from datasets import load_dataset
import pandas as pd

def load_data():
    ds = load_dataset("mteb/tweet_sentiment_extraction")

    train_df=ds["train"]
    test_df=ds["test"]

    train_df=pd.DataFrame(train_df)
    test_df=pd.DataFrame(test_df)

    final_df=pd.concat([test_df,train_df],ignore_index=True)
    print(f"shape of the dats is:{final_df.shape}")
    final_df.to_parquet("train.parquet", compression="snappy")
    print("Data saved to parquet")
    # final_df.to_csv("train.csv", index=False)

if __name__=="__main__":
    load_data()
