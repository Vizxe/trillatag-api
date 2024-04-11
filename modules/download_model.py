from huggingface_hub import hf_hub_download

def download(repo_id, filename):
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return path
