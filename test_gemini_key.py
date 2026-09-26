import os
import sys

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("❌ GEMINI_API_KEY belum di-set.")
    sys.exit(1)

try:
    from google import genai
except ImportError:
    print("❌ Library belum terinstall.")
    print("   Jalankan: pip install -U google-genai --break-system-packages")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

# Berdasarkan Rate Limit dashboard kamu, project ini punya quota free tier
# untuk Gemini 2.5 Flash (5 RPM / 250K TPM). Test model ini duluan.
MODELS_TO_TRY = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]

print("== Test request langsung (SDK baru) ==")
for model_name in MODELS_TO_TRY:
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents="Balas dengan satu kata: OK"
        )
        print(f"✅ {model_name} berhasil respon: {resp.text.strip()}")
        print(f"\n>> Pakai model ini di Cline: {model_name}")
        break
    except Exception as e:
        print(f"❌ {model_name} gagal: {e}")
else:
    print("\n⚠️  Semua model gagal dengan SDK baru juga.")
    print("   Cek lagi di AI Studio > Rate Limit apakah quota masih tersedia (5 RPM tadi).")