from huggingface_hub import HfApi

api = HfApi()

api.upload_large_folder(
    folder_path="/home/wilfred/sys/analytica/uroflow-thesis/Data_Flow_JBHI",
    repo_id="wilfredk/raw_2",
    repo_type="dataset"
)

