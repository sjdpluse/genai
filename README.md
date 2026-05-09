# 🤖 ApexTrade ML Signal System
### سیستم تولید سیگنال با یادگیری ماشین

---

## معماری سیستم

```
Binance API → Feature Engineering → Random Forest ML → FastAPI → Supabase → وب‌سایت
                                                            ↑
                                              Win/Loss Tracker (هر ۱ ساعت)
```

---

## ساختار فایل‌ها

```
apextrade_signal/
├── main.py              # FastAPI — تمام endpoint‌ها
├── config.py            # تنظیمات مرکزی
├── data_fetcher.py      # دریافت OHLCV از Binance
├── feature_engineer.py  # محاسبه اندیکاتورها + ویژگی‌سازی
├── ml_model.py          # آموزش، ذخیره، و پیش‌بینی مدل
├── signal_generator.py  # تولید سیگنال نهایی + مدیریت ریسک
├── win_tracker.py       # ردیابی خودکار win/loss
├── supabase_client.py   # اتصال به Supabase
├── requirements.txt
├── Procfile             # برای Railway
└── .env.example
```

---

## قدم اول: نصب محلی و تست

### ۱. Python نصب کنید (نسخه ۳.۱۱+)
از https://python.org/downloads دانلود کنید

### ۲. پوشه پروژه را بسازید
```bash
mkdir apextrade_signal
cd apextrade_signal
```

### ۳. فایل‌های پروژه را در این پوشه بگذارید

### ۴. Virtual Environment بسازید
```bash
# Windows:
python -m venv venv
venv\Scripts\activate

# Mac/Linux:
python3 -m venv venv
source venv/bin/activate
```

### ۵. کتابخانه‌ها را نصب کنید
```bash
pip install -r requirements.txt
```

### ۶. فایل .env بسازید
```
فایل .env.example را کپی کنید و نامش را .env بگذارید
سپس مقادیر Supabase را پر کنید:
```

#### چطور SUPABASE_KEY را پیدا کنیم؟
1. وارد supabase.com شوید
2. پروژه‌تان را باز کنید
3. Settings → API
4. **service_role** key را کپی کنید (نه anon key!)
5. در .env بگذارید

### ۷. آموزش اولیه مدل
```bash
python -c "from signal_generator import run_training; run_training()"
```
⏱ این ۳-۵ دقیقه طول می‌کشد — صبر کنید

### ۸. سرور را اجرا کنید
```bash
uvicorn main:app --reload
```

### ۹. تست کنید
مرورگر را باز کنید:
- http://localhost:8000/signal → سیگنال فعلی
- http://localhost:8000/performance → آمار عملکرد
- http://localhost:8000/health → وضعیت سرور
- http://localhost:8000/docs → مستندات API (Swagger)

---

## قدم دوم: Deploy روی Railway (رایگان)

Railway.app یک پلتفرم رایگان برای اجرای Python سرور است.

### ۱. حساب Railway بسازید
به https://railway.app بروید و با GitHub وارد شوید

### ۲. پروژه را روی GitHub بگذارید
```bash
git init
git add .
git commit -m "ApexTrade ML Signal System"
# یک repo جدید در github.com بسازید و push کنید
git remote add origin https://github.com/USERNAME/apextrade-signal.git
git push -u origin main
```

> ⚠️ مطمئن شوید فایل .env را commit نکنید!
> فایل `.gitignore` بسازید و این را در آن بگذارید:
> ```
> .env
> model/
> __pycache__/
> venv/
> ```

### ۳. در Railway deploy کنید
1. New Project → Deploy from GitHub repo
2. Repo خود را انتخاب کنید
3. Railway خودکار `Procfile` را می‌خواند و شروع می‌کند

### ۴. Environment Variables را اضافه کنید
در Railway Dashboard:
- Variables → Add Variable
```
SUPABASE_URL = https://rjhttcokuwlhggyqqxph.supabase.co
SUPABASE_KEY = YOUR_SERVICE_ROLE_KEY
```

### ۵. URL سرویس را بگیرید
Railway یک URL مثل این می‌دهد:
`https://apextrade-signal-production.up.railway.app`

---

## قدم سوم: اتصال به وب‌سایت

در فایل `index.html` وب‌سایت‌تان، بخش سیگنال را عوض کنید:

```javascript
// آدرس سرور Python خود را اینجا بگذارید
const ML_SIGNAL_API = "https://YOUR-RAILWAY-URL.up.railway.app";

async function generateAISignal() {
    const card = document.getElementById('signalCard');
    card.innerHTML = `<div class="signal-header wait">در حال تحلیل ML...</div>`;

    try {
        const resp = await fetch(`${ML_SIGNAL_API}/signal`);
        const data = await resp.json();
        const sig  = data.signal;

        const typeClass = sig.type === 'LONG' ? 'long' : sig.type === 'SHORT' ? 'short' : 'wait';
        
        card.innerHTML = `
        <div class="signal-header ${typeClass}">
            <div class="signal-type-badge">${sig.type}</div>
            <div class="signal-meta">
                <div class="signal-symbol">BTCUSDT · ML Model</div>
                <div class="signal-confidence">
                    <i class="fas fa-brain"></i> Confidence: ${sig.confidence}%
                </div>
            </div>
        </div>
        <div class="signal-body">
            ${sig.type !== 'WAIT' ? `
            <div class="signal-grid">
                <div class="signal-item">
                    <div class="signal-item-label">Entry</div>
                    <div class="signal-item-val">${sig.entry_price}</div>
                </div>
                <div class="signal-item">
                    <div class="signal-item-label">Stop Loss</div>
                    <div class="signal-item-val short">${sig.stop_loss}</div>
                </div>
                <div class="signal-item">
                    <div class="signal-item-label">Take Profit 1</div>
                    <div class="signal-item-val long">${sig.take_profit1}</div>
                </div>
                <div class="signal-item">
                    <div class="signal-item-label">Take Profit 2</div>
                    <div class="signal-item-val long">${sig.take_profit2}</div>
                </div>
            </div>` : `
            <div style="text-align:center;padding:16px;color:#f59e0b;">
                <i class="fas fa-pause-circle" style="font-size:2rem;display:block;margin-bottom:8px"></i>
                شرایط ورود مناسب نیست
            </div>`}
            <div class="signal-reason">
                <strong>تحلیل ML:</strong><br>${sig.reasons}
            </div>
        </div>`;

    } catch(e) {
        card.innerHTML = `<div class="signal-header wait">خطا در دریافت سیگنال ML</div>`;
    }
}
```

---

## آموزش مجدد مدل

هر ماه یک بار مدل را با داده جدید آموزش دهید:
```bash
# از طریق API (بدون نیاز به دسترسی به سرور):
curl -X POST https://YOUR-RAILWAY-URL.up.railway.app/train
```

---

## API Endpoints

| Endpoint | Method | توضیح |
|----------|--------|-------|
| `/signal` | GET | سیگنال فعلی با ML |
| `/train` | POST | آموزش مجدد مدل |
| `/track` | GET | ردیابی دستی win/loss |
| `/performance` | GET | آمار عملکرد کلی |
| `/history?limit=20` | GET | تاریخچه سیگنال‌ها |
| `/health` | GET | وضعیت سرور |
| `/docs` | GET | مستندات Swagger |

---

## نکات مهم

### چرا Random Forest؟
- بدون overfitting زیاد
- نیازی به نرمال‌سازی دقیق ندارد
- feature importance داخلی دارد
- روی این حجم داده سریع است
- نتایج قابل تفسیر هستند

### Walk-Forward Validation چیست؟
مدل هرگز از داده آینده برای آموزش استفاده نمی‌کند.
در هر fold، آموزش فقط روی گذشته و ارزیابی روی آینده است.
این از lookahead bias (باگ رایج backtesting) جلوگیری می‌کند.

### اعتبار confidence
سیگنال‌هایی که مدل با confidence بالاتر از ۶۵٪ صادر می‌کند
معمولاً win rate بهتری دارند. این را از `/performance` می‌توانید ببینید.

---

## مشکل‌یابی

**خطا: No module named 'ta'**
```bash
pip install ta
```

**خطا: SUPABASE_KEY**
مطمئن شوید service_role key را گذاشتید، نه anon key.

**مدل آموزش نمی‌بیند**
Binance ممکن است در برخی مناطق محدود باشد.
از VPN استفاده کنید یا سرور را روی Railway اجرا کنید.

---

ساخته‌شده برای ApexTrade | سجاد
