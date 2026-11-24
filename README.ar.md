# ميزان — Mizan

**وزّن الـ fine-tune حقك قبل ما تشغّله.**

[![CI](https://github.com/XenzV/mizan/actions/workflows/ci.yml/badge.svg)](https://github.com/XenzV/mizan/actions/workflows/ci.yml) [![Release](https://img.shields.io/github/v/release/XenzV/mizan)](https://github.com/XenzV/mizan/releases) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ميزان أداة سطر أوامر صغيرة تحسب الذاكرة القصوى المطلوبة لتدريب LoRA / QLoRA من أرقام معمارية الموديل ومن ملف الإعدادات. تشتغل في أجزاء من الثانية، ما تحتاج PyTorch ولا CUDA، وما تنزّل أوزان ولا تتصل بأي سيرفر.

[English](README.md) · [الطريقة](docs/METHOD.md)

## المشكلة

تكتب كونفيغ، تشغّل التدريب، تمشي، وترجع بعد 40 دقيقة تلاقي `CUDA out of memory` عند الخطوة 3. الكونفيغ ما فيه خطأ نحوي — بس ببساطة ما اتسع في الكرت.

ميزان يجاوب على سؤال واحد قبل ما تحرق وقت الـ GPU: **هل هذا الإعداد بيسع؟ وإذا لا، وش اللي لازم يتغيّر؟**

## التثبيت

```bash
git clone https://github.com/XenzV/mizan.git
cd mizan
pip install -e .
mizan --help
```

يحتاج Python 3.10 فأعلى. بدون تبعيات وقت التشغيل. ملفات YAML تشتغل بالparser المدمج، وإذا تبي YAML كامل ثبّت `pip install -e ".[yaml]"`.

## الاستخدام

```bash
mizan weigh -m llama-3.1-8b --gpu 8 --seq 512
mizan weigh --config soup.yaml --gpu 8
mizan fit -m qwen2.5-7b --gpu 8 --seq 2048
mizan check --config soup.yaml --strict
mizan doctor
mizan presets
mizan init -m llama-3.1-8b --gpu 8 --out mizan.yaml
mizan report --config soup.yaml --gpu 8 --out preflight.md
```

`--model` يقبل اسم جاهز، أو اختصار (`llama-8b`, `qwen-7b`)، أو معرّف Hugging Face مثل `meta-llama/Llama-3.1-8B-Instruct`، أو مجلد محلي فيه `config.json`. تقدر كمان تمرّر أبعاد المعمارية بنفسك.

### weigh

يعطيك تفصيل كل مكوّن من الذاكرة بالـ GiB مع شريط نسبة، ثم الحكم النهائي: `OK` أو `TIGHT` أو `OVER`. أضف `--json` لتطلع نتيجة قابلة للقراءة برمجياً في الـ CI.

مثال حقيقي على Llama-3.1-8B، كمّية 4bit، طول 512، وحفظ تدرّج مفعّل:

```
ESTIMATED PEAK         6.50 GiB
VERDICT  OK  (8.0 GiB GPU)
       ~1.10 GiB of headroom under the 5% safety line
```

### fit

يجرب تركيبات من الكمّية وحجم الـ micro-batch وحفظ التدرّج، ويرجّع أكبر إعداد يبقى تحت `--target-util` (الافتراضي 90%) مع مقطع كونفيغ جاهز للصق.

```
$ mizan fit -m llama-3.1-8b --gpu 8 --seq 512
  Recommended: 4bit, micro-batch 2, grad-accum 8, ckpt on
  Estimated peak 6.95 GiB
```

### check

يفحص الكونفيغ: مفاتيح غير معروفة، بيانات ناقصة، أرقام غير صالحة، أهداف LoRA غلط، وكمّيات خاطئة. يرجّع رمز خروج `1` عند الخطأ، و`--strict` يحوّل التحذيرات لأخطاء. مصمم للـ CI.

## كيف تحسب الأرقام

| المكوّن | اللي يحسبه |
| --- | --- |
| `base_weights` | عدد المعاملات × بايت لكل معامل (NF4 = 0.5625، int8 = 1.0625، bf16 = 2) |
| `lora_weights` | الرتبة × (دخل + خرج) لكل طبقة مستهدفة × عدد الطبقات |
| `trainable_states` | أوزان + تدرّجات + حالة الـ optimizer (AdamW: 12 بايت/معامل) |
| `activations` | التنسورات المحفوظة + إعادة حساب طبقة واحدة عند تفعيل checkpointing + مصفوفة الـ logits |
| `cuda_context` | بدل ثابت للـ runtime، الافتراضي 700 MiB |
| `fragmentation` | 8% من المجموع الفرعي افتراضياً |

عدد المعاملات يُشتق من المعمارية، ما ينزّل checkpoint. في Llama-3.1-8B طلع 8.03B، وLoRA رتبة 16 على كل الطبقات طلع 41.9M معامل قابل للتدريب. التفاصيل في [docs/METHOD.md](docs/METHOD.md).

## حدود واضحة

- هذا **تقدير تحليلي** مو profiler. توقّع فرق تقريباً ±20%. أكّد بتشغيل 20 خطوة قبل أي مهمة طويلة.
- يغطي QLoRA/LoRA الاعتيادي. ما يغطي layer streaming ولا offload للـ CPU/NVMe ولا DeepSpeed ZeRO ولا التوازي.
- نماذج MoE وفك التشفير المت speculative خارج النطاق.
- `--headroom` سياسة حماية مو ثابت فيزيائي؛ الافتراضي 5%.

## ميزان مقابل أداة تدريب

ميزان ما يدرّب نماذج. أداة التدريب تحتاج تجاوب على سؤال ميزان أول. مسار شائع مع Soup:

```bash
mizan check --config soup.yaml --strict
mizan weigh --config soup.yaml --gpu 8
mizan fit   --config soup.yaml --gpu 8 --seq 1024
soup train  --config soup.yaml
```

MIT. التطوير: `pip install -e ".[dev]"` ثم `pytest -q`.
