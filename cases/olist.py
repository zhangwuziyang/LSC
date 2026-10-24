"""Olist case study: stacked seller x treatment-month panel in the three-window design.

For each treatment month T (default 2017-06 .. 2018-05) and each seller with at least
`min_orders` delivered orders in T and some sales in T-3..T-1:

    dose D        mean delivery days (purchase -> delivered) of the seller's orders in T
    covariates    measured in T-3..T-2 (pre1), so that the placebo window T-1 is clean
    dY            daily GMV in T+1..T+3 minus daily GMV in T-3..T-1
    dY_placebo    daily GMV in T-1 minus daily GMV in T-3..T-2
    group         seller id (sellers appear in several months; resampling is by seller)

Data: Olist Brazilian e-commerce public dataset (MIT licence), e.g.
    git clone --depth 1 https://github.com/olist/work-at-olist-data
"""
from pathlib import Path

import numpy as np
import pandas as pd

COVARIATES = ["log_pre_gmv", "pre_orders", "pre_days", "has_pre_days", "pre_same_state", "pre_price",
              "pre_freight_ratio", "seller_state", "category", "tenure_months", "month_index"]


def _load(path):
    p = Path(path)
    o = pd.read_csv(p / "olist_orders_dataset.csv",
                    parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"])
    o = o[o.order_status == "delivered"].dropna(subset=["order_delivered_customer_date"])
    o["days"] = (o.order_delivered_customer_date - o.order_purchase_timestamp).dt.total_seconds() / 86400
    it = pd.read_csv(p / "olist_order_items_dataset.csv")
    cu = pd.read_csv(p / "olist_customers_dataset.csv")[["customer_id", "customer_state"]]
    se = pd.read_csv(p / "olist_sellers_dataset.csv")[["seller_id", "seller_state"]]
    pr = pd.read_csv(p / "olist_products_dataset.csv")[["product_id", "product_category_name"]]
    x = (it.merge(o[["order_id", "customer_id", "order_purchase_timestamp", "days"]], on="order_id")
           .merge(cu, on="customer_id").merge(se, on="seller_id").merge(pr, on="product_id", how="left"))
    x["month"] = x.order_purchase_timestamp.dt.to_period("M")
    x["same_state"] = (x.customer_state == x.seller_state).astype(float)
    return x


def _window(x, lo, hi):
    return x[(x.month >= lo) & (x.month <= hi)]


def _ndays(lo, hi):
    return sum(m.days_in_month for m in pd.period_range(lo, hi, freq="M"))


def build_panel(path, start="2017-06", end="2018-05", min_orders=3, winsor=0.01):
    x = _load(path)
    state_codes = {s: i for i, s in enumerate(sorted(x.seller_state.unique()))}
    cat_codes = {c: i for i, c in enumerate(sorted(x.product_category_name.dropna().unique()))}
    first_sale = x.groupby("seller_id").month.min()
    rows = []
    for k, T in enumerate(pd.period_range(start, end, freq="M")):
        tr = x[x.month == T]
        orders_T = tr.groupby("seller_id").order_id.nunique()
        pre = _window(x, T - 3, T - 1)
        sellers = orders_T[(orders_T >= min_orders) & orders_T.index.isin(pre.seller_id.unique())].index
        if len(sellers) == 0:
            continue
        # dose: order-level mean delivery days in T
        dose = tr[tr.seller_id.isin(sellers)].drop_duplicates(["seller_id", "order_id"]).groupby("seller_id").days.mean()
        pre1 = _window(x, T - 3, T - 2)
        g1 = pre1[pre1.seller_id.isin(sellers)].groupby("seller_id")
        f = pd.DataFrame(index=sellers)
        f["pre_gmv1"] = g1.price.sum()
        f["pre_orders"] = g1.order_id.nunique()
        f["pre_days"] = pre1.drop_duplicates(["seller_id", "order_id"]).groupby("seller_id").days.mean()
        f["pre_same_state"] = g1.same_state.mean()
        f["pre_price"] = g1.price.mean()
        f["pre_freight_ratio"] = (g1.freight_value.sum() / g1.price.sum())
        main_cat = pre[pre.seller_id.isin(sellers)].groupby("seller_id").product_category_name.agg(
            lambda s: s.mode().iat[0] if s.notna().any() else None)
        gmv = lambda lo, hi: _window(x, lo, hi).groupby("seller_id").price.sum().reindex(sellers).fillna(0.0)
        y_pre = gmv(T - 3, T - 1) / _ndays(T - 3, T - 1)
        y_post = gmv(T + 1, T + 3) / _ndays(T + 1, T + 3)
        y_pre1 = gmv(T - 3, T - 2) / _ndays(T - 3, T - 2)
        y_pre2 = gmv(T - 1, T - 1) / _ndays(T - 1, T - 1)
        f = f.fillna({"pre_gmv1": 0.0, "pre_orders": 0.0})
        out = pd.DataFrame({
            "seller_id": sellers, "T": str(T), "month_index": k, "D": dose.reindex(sellers).values,
            "log_pre_gmv": np.log1p(f.pre_gmv1 / _ndays(T - 3, T - 2)).values,
            "pre_orders": f.pre_orders.values,
            "has_pre_days": f.pre_days.notna().astype(float).values,
            "pre_days": f.pre_days.values,
            "pre_same_state": f.pre_same_state.values, "pre_price": f.pre_price.values,
            "pre_freight_ratio": f.pre_freight_ratio.values,
            "seller_state": [state_codes[s] for s in x.drop_duplicates("seller_id").set_index("seller_id")
                             .seller_state.reindex(sellers)],
            "category": [cat_codes.get(c, -1) for c in main_cat.reindex(sellers)],
            "tenure_months": [(T - first_sale[s]).n for s in sellers],
            "dY": (y_post - y_pre).values, "dY_placebo": (y_pre2 - y_pre1).values,
            "y_pre": y_pre.values, "orders_T": orders_T.reindex(sellers).values,
        })
        rows.append(out)
    df = pd.concat(rows, ignore_index=True)
    # Sellers with no orders in pre1 have no pre-period delivery or price data: fill with
    # the median and keep the has_pre_days flag.
    for c in ["pre_days", "pre_same_state", "pre_price", "pre_freight_ratio"]:
        df[c] = df[c].fillna(df[c].median())
    if winsor:
        for c in ["dY", "dY_placebo"]:
            lo, hi = df[c].quantile([winsor, 1 - winsor])
            df[c] = df[c].clip(lo, hi)
    return df
