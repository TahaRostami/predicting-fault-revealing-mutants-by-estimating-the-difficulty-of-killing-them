import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error,mean_absolute_error

# region evaluator(s)
def recall_at_k(df,actual_class,pred_class,k,if_boundaries_violated="default",is_sorted=False):
    if k>len(df) or k<1:
        k=len(df) if k>len(df) else 1
        if if_boundaries_violated=="exception":raise Exception()
    inner_df = df.sort_values(by=[pred_class], ascending=False) if is_sorted==False else df
    cnt_relevant_items =inner_df[actual_class].sum()
    cnt_recommendations_that_are_relevant=inner_df.head(k)[actual_class].sum()
    return cnt_recommendations_that_are_relevant/cnt_relevant_items
def precision_at_k(df,actual_class,pred_class,k,if_boundaries_violated="default",add_min_to_formula=False,is_sorted=False):
    if k>len(df) or k<1:
        k=len(df) if k>len(df) else 1
        if if_boundaries_violated=="exception":raise Exception()
    inner_df = df.sort_values(by=[pred_class], ascending=False) if is_sorted==False else df
    cnt_recc=k
    cnt_recommendations_that_are_relevant=inner_df.head(k)[actual_class].sum()
    if add_min_to_formula:
        cnt_recc=min(inner_df[actual_class].sum(),cnt_recc)
    return cnt_recommendations_that_are_relevant/cnt_recc
def f1_at_k(recall_at_k,precision_at_k):
    return 2*precision_at_k*recall_at_k/(precision_at_k+recall_at_k)
def eval_overall(df,actual_class,pred_class):
    overall_roc_auc = roc_auc_score(df[actual_class], df[pred_class])

    recall_at_5_prec = recall_at_k(df, actual_class, pred_class, round(len(df) * 0.05))
    recall_at_10_prec = recall_at_k(df, actual_class, pred_class, round(len(df) * 0.1))
    recall_at_20_prec=recall_at_k(df,actual_class,pred_class,round(len(df)*0.2))

    precision_at_5_prec = precision_at_k(df, actual_class, pred_class, round(len(df) * 0.05))
    precision_at_10_prec = precision_at_k(df, actual_class, pred_class, round(len(df) * 0.1))
    precision_at_20_prec = precision_at_k(df, actual_class, pred_class, round(len(df) * 0.2))

    f1_at_5_prec = f1_at_k(recall_at_5_prec,precision_at_5_prec)
    f1_at_10_prec = f1_at_k(recall_at_10_prec,precision_at_10_prec)
    f1_at_20_prec = f1_at_k(recall_at_20_prec,precision_at_20_prec)

    return {"roc-auc":overall_roc_auc,
            "recall@5%":recall_at_5_prec,"recall@10%":recall_at_10_prec,"recall@20%":recall_at_20_prec,
            "prec.@5%":precision_at_5_prec,"prec.@10%":precision_at_10_prec,"prec.@20%":precision_at_20_prec,
            "f1@5%":f1_at_5_prec,"f1@10%":f1_at_10_prec,"f1@20%":f1_at_20_prec}
def eval_reg(df,actual_val,estimate_val):
    return {"MSE":mean_squared_error(df[actual_val],df[estimate_val],squared=True),
            "RMSE":mean_squared_error(df[actual_val],df[estimate_val],squared=False),
            "MAE":mean_absolute_error(df[actual_val],df[estimate_val])}
# endregion

# region report(s)
def report_1(filename_results,engine="pyarrow"):
    df_results = pd.read_parquet(filename_results, engine=engine)
    for ds_name, df_dataset in df_results.groupby("ds_name"):
        res_proposed = eval_overall(df_dataset, "actual_class", "model_prediction")
        data = {item: [] for item in ["method_name"] + list(res_proposed.keys())}
        data["method_name"] = ["Proposed"]
        for metric in res_proposed.keys():
            data[metric].append(f"{res_proposed[metric]:.4f}")
        print(f"Dataset: {ds_name}")
        print(pd.DataFrame(data).to_markdown(index=False))
        print("*" * 145, '\n')
def report_2(filename_results,engine="pyarrow"):
    df_results = pd.read_parquet(filename_results, engine=engine)
    df_results=df_results.dropna()
    for ds_name, df_dataset in df_results.groupby("ds_name"):
        res_proposed = eval_reg(df_dataset, 'actual_SM_kill_freq','estimated_SM_kill_freq')
        data = {item: [] for item in ["method_name"] + list(res_proposed.keys())}
        data["method_name"] = ["Proposed"]
        for metric in res_proposed.keys():
            data[metric].append(f"{res_proposed[metric]:.4f}")
        print(f"Dataset: {ds_name}")
        print(pd.DataFrame(data).to_markdown(index=False))
        print("*" * 145, '\n')
# endregion

if __name__ == "__main__":

    print("Classification")
    report_1("../results/main_results.parquet", engine="pyarrow")
    print("\nRegression")
    report_2("../results/main_results.parquet",engine="pyarrow")