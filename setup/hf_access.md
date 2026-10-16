# Hugging Face access (do this first)

1. Log in on huggingface.co, open https://huggingface.co/CohereLabs/tiny-aya-global and accept the
   gated terms (CC-BY-NC plus Cohere Acceptable Use Policy). Repeat for tiny-aya-base-32K and
   tiny-aya-fire. Approval is usually immediate.
2. Create a read token at https://huggingface.co/settings/tokens.
3. On each machine:
   ```bash
   pip install -U huggingface_hub hf_transfer
   hf auth login            # paste the token
   export HF_HUB_ENABLE_HF_TRANSFER=1
   hf download CohereLabs/tiny-aya-global --local-dir models/tiny-aya-global
   hf download CohereLabs/tiny-aya-global-GGUF --local-dir models/tiny-aya-global-GGUF   # official GGUFs, for comparison
   ```
4. Datasets you will need later (accept terms where gated): CohereLabs/Global-MMLU,
   openlanguagedata/flores_plus, CohereLabs/aya_dataset, CohereLabs/aya_collection,
   CohereLabs/aya_evaluation_suite (Dolly-200 prompts for the judge and human eval).
