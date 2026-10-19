# OpenBLAS Matrix Multiplication Benchmark

معيار لمضاعفة المصفوفات (SGEMM) باستخدام **OpenBLAS** على المعالج.

## الملفات
- `sgemm_bench.c` — برنامج C يحسب `C = A × B` عبر `cblas_sgemm` ويقيس GFLOPS لأحجام مربّعة متزايدة (64 → 4096).
- `.github/workflows/matmul.yml` — GitHub Action عامة (public) تشغّل المعيار على `ubuntu-latest`.

## التشغيل محليًا
```bash
# تثبيت OpenBLAS (Debian/Ubuntu)
sudo apt-get install -y libopenblas-dev gcc

# البناء
gcc -O3 -march=native -o sgemm_bench sgemm_bench.c -lopenblas -lpthread -lm

# التشغيل (الملتي-ثريد)
./sgemm_bench 4096

# التشغيل (ثريد واحد)
OPENBLAS_NUM_THREADS=1 ./sgemm_bench 4096
```

## GitHub Actions
1. ارفع المستودع إلى GitHub (سواء public أو خاص).
2. الانتقال إلى تبويب **Actions** → شغّل `OpenBLAS Matrix-Multiply Benchmark` يدويًا عبر **Run workflow**، أو ادفع أي تعديل على `sgemm_bench.c`.
3. النتائج (CPU + GFLOPS) تظهر في سجلّ الخطوة `Run benchmark`.

### ملاحظة عن الأداء
- **المضاعفة الفعلية**: `2 × N³` عملية فاصلة عائمة (FLOPS).
- GFLOPS = `2 × N³ ÷ الزمن`.
- `-march=native` يفعّل AVX/AVX-512 حسب معالج الـ runner.
- عدد الثريدات يتحكّم فيه `OPENBLAS_NUM_THREADS`.
