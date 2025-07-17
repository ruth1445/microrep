import os
print("📍 Current directory:", os.getcwd())

try:
    from router import Router
    print("✅ Router imported successfully!")
except Exception as e:
    print("❌ Import failed:", e)

