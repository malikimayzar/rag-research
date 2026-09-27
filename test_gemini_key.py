import os

MODELS_TO_TRY = ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest")

def check_gemini_connection(api_key: str, model_names=MODELS_TO_TRY) -> str | None:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Library google-genai belum terinstall. Jalankan "
            "./venv/bin/python -m pip install google-genai."
        ) from exc

    client = genai.Client(api_key=api_key)
    for model_name in model_names:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents="Balas dengan satu kata: OK",
            )
        except Exception as exc:
            print(f"{model_name} gagal: {exc}")
            continue

        response_text = (response.text or "").strip()
        if not response_text:
            print(f"{model_name} memberikan respons kosong.")
            continue

        print(f"{model_name} berhasil merespons: {response_text}")
        print(f"Model yang dapat dipakai: {model_name}")
        return model_name

    print("Semua model gagal. Periksa akses API dan kuota Gemini.")
    return None

def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY belum di-set.")
        return 1

    try:
        working_model = check_gemini_connection(api_key)
    except RuntimeError as exc:
        print(exc)
        return 2
    return 0 if working_model else 1

if __name__ == "__main__":
    raise SystemExit(main())