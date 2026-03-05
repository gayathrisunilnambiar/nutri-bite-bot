import pandas as pd
df = pd.read_csv(r'c:\Gayathri\BITE_BOT\nutri-bite-bot\data\ifct_2017_nutritional_values.csv')
egg = df[df['ingredient'].str.lower().str.contains('egg')]
print('Egg-related ingredients:')
for _, r in egg.iterrows():
    prot = r.get('protein_g_per_100g', 0)
    phos = r.get('phosphorus_mg_per_100g', 0)
    ratio = phos / prot if prot > 0 else 0
    print(f"  {r['ingredient']}: protein={prot}, phos={phos}, ratio={ratio:.1f}")

print()
high_prot = df[df['protein_g_per_100g'] > 5].copy()
high_prot['phos_ratio'] = high_prot['phosphorus_mg_per_100g'] / high_prot['protein_g_per_100g']
high_prot = high_prot.sort_values('phos_ratio')
print('Top 15 lowest phos/protein ratio:')
for _, r in high_prot.head(15).iterrows():
    print(f"  {r['ingredient']:35s} prot={r['protein_g_per_100g']:5.1f}g  phos={r['phosphorus_mg_per_100g']:6.1f}mg  ratio={r['phos_ratio']:5.1f}")

print()
print('Bottom 10 highest phos/protein ratio:')
for _, r in high_prot.tail(10).iterrows():
    print(f"  {r['ingredient']:35s} prot={r['protein_g_per_100g']:5.1f}g  phos={r['phosphorus_mg_per_100g']:6.1f}mg  ratio={r['phos_ratio']:5.1f}")
