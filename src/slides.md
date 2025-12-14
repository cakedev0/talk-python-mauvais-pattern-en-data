---
title: "Mauvais patterns en data science Python"
---
<style>
.reveal pre code {
  max-height: 95vh;
  overflow: auto;
</style>

# Mauvais patterns en data science Python
### Lighting talk

14 décembre 2025

---

# Quel est le point commun de ces 3 fonctions?

-v-

```python
def merge_dicts(list_of_dicts: list[dict]) -> dict:
    return reduce(lambda d1, d2: {**d1, **d2}, list_of_dicts)
```

-v-

```python
def compute_n_uniques_by_user(df: pd.DataFrame) -> dict:
    n_uniques = {}
    for user in df['user'].unique():
        user_activity = df[df['user'] == user]
        n_uniques[user] = user_activity['item'].nunique()
    return n_uniques
```

-v-

```python
def compute_avg_by_group(group_idx: np.ndarray[int], x: np.ndarray) -> np.ndarray:
    n_groups = group_idx.max() + 1
    avg_by_group = np.empty(n_groups)
    for idx in range(n_groups):
        avg_by_group[idx] = np.mean(x[group_idx == idx])
    return avg_by_group
```

-v-

## Indice

```python
n = sum(len(d) for d in list_of_dicts)
n = len(df)
n = len(x)
```

---

# Réponse: Inefficacité algorithmique!

Ces 3 algos peuvent être en $ O(n^2) $

-v-

## Des patterns courants

- Dictionnaires : vu tel quel en production
- Pandas : le plus fréquent chez les data scientists<!-- .element: class="fragment" -->
- Numpy : vu dans scikit-learn !<!-- .element: class="fragment" -->

---

# Pourquoi on écrit ça?

La complexité algorithmique est cachée par la syntaxe

-v-

```python
for user in df['user'].unique():
    user_activity = df[df['user'] == user]  # ← Process TOUTES les lignes!
    ...
```

Perçu comme $ O(n) $ alors que c'est $ O(n^2) $

---

# Comment éviter ça?

1. Comprendre Python et ses librairies
2. Connaître les bons patterns<!-- .element: class="fragment" -->

---

# Solution dictionnaires

```python [|2-3]
merged_dict = {}
for d in list_of_dicts:
    merged_dict.update(d)  # O(len(d))
```

---

# Solution pandas

```python
n_uniques_by_user = df.groupby('user')['item'].nunique()
```

-v-

## Si c'est plus compliqué?

```python
f = lambda user_activity: user_activity['item'].nunique()

df.groupby('user').apply(f, include_groups=False)
```

Plus lent (overhead par groupe) mais $ O(n) $ !

---

# Solution numpy

```python [|1-2|4]
n_by_group = np.bincount(group_idx, minlength=n_groups)
sum_by_group = np.bincount(group_idx, weights=x, minlength=n_groups)

avg_by_group = sum_by_group / n_by_group
```

-v-

## Autre opération: minimum

```python
min_by_group = np.full(n_groups, np.inf)
np.minimum.at(min_by_group, group_idx, x)
```

Les ufuncs de numpy: `.at`, `.reduce`, `.accumulate`

---

# Merci !

---

# Pour aller plus loin

---

# 1. IDs non-denses

```python
x = np.random.rand(n)
group_id = np.random.choice(np.random.choice(10**18, n_groups), n)
```

-v-

```python [|1|2|5]
unique_ids, group_idx = np.unique(group_id, return_inverse=True)
# O(n log n), mais 10-100x plus lent que bincount en pratique
n_groups = unique_ids.size

sum_by_group = np.bincount(group_idx, weights=x)
```

---

# 2. Si les ufuncs ne suffisent pas

Exemple: médiane par groupe

-v-

## Utiliser scipy sparse

```python [|1-5|7-14]
x_per_group = csr_array((
    x,
    (group_idx, np.arange(n))
), shape=(n_groups, n))

def median_per_row(data, indptr):
    n_rows = indptr.size - 1
    out = np.full(n_rows, np.nan)
    for row in range(n_rows):
        row_start, row_end = indptr[row], indptr[row + 1]
        if row_start == row_end:
            continue
        out[row] = np.median(data[row_start:row_end])
    return out
```

-v-

## Pour la vitesse: numba

```python
from numba import njit

median_per_row_numba = njit(median_per_row)

# ou Cython dans scikit-learn
```

---

# Merci !