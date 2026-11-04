from datasets import load_dataset
from collections import Counter

ds = load_dataset("LabHC/bias_in_bios")
print(ds)  # train / dev / test, kolommen: hard_text, profession, gender

professions = [
    "accountant", "architect", "attorney", "chiropractor", "comedian", "composer",
    "dentist", "dietitian", "dj", "filmmaker", "interior_designer", "journalist",
    "model", "nurse", "painter", "paralegal", "pastor", "personal_trainer",
    "photographer", "physician", "poet", "professor", "psychologist", "rapper",
    "software_engineer", "surgeon", "teacher", "yoga_teacher",
]
genders = ["male", "female"]

train = ds["train"]

# 1. Lege teksten?
for split in ds:
    print(split, "lege teksten:", sum(1 for t in ds[split]["hard_text"] if not t or not t.strip()))

# 2. Verdeling van beroepen (train)
prof_counts = Counter(train["profession"])
for idx, count in prof_counts.most_common():
    print(f"{professions[idx]:18s} {count:7d}  ({count/len(train):.1%})")

# 3. Verdeling van gender
gender_counts = Counter(train["gender"])
print({genders[k]: f"{v} ({v/len(train):.1%})" for k, v in gender_counts.items()})

# 4. Aandeel vrouwen per beroep (het belangrijkste voor de bias-analyse)
female_per_prof = Counter(p for p, g in zip(train["profession"], train["gender"]) if g == 1)
print("\nAandeel vrouwen per beroep:")
for idx in sorted(prof_counts, key=lambda i: female_per_prof[i] / prof_counts[i]):
    print(f"{professions[idx]:18s} {female_per_prof[idx]/prof_counts[idx]:.1%}")

# 5. Tekstlengte in woorden
lengths = [len(t.split()) for t in train["hard_text"]]
print(f"\nGemiddeld {sum(lengths)/len(lengths):.1f} woorden, min {min(lengths)}, max {max(lengths)}")

# 6. Verdeling per split vergelijken (zijn de splits vergelijkbaar?)
for split in ds:
    c = Counter(ds[split]["profession"])
    top3 = [(professions[i], f"{n/len(ds[split]):.1%}") for i, n in c.most_common(3)]
    print(split, top3)

# 7. Voorbeelden
for ex in train.shuffle(seed=42).select(range(5)):
    print(f"[{professions[ex['profession']]}, {genders[ex['gender']]}] {ex['hard_text'][:200]}")