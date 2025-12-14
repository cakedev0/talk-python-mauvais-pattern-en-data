---
title: "Mauvais patterns en data science Python"
---
<style>
.reveal pre code {
  max-height: 95vh;
  overflow: auto;
</style>

## Un mauvais patterns en Python data et comment l'éviter
#### Lighting talk

16 décembre 2025 - Présenté à  [laturbine.coop](https://turbine.coop/) par Arthur Lacote

[Source des slides et notebook sur Github](https://github.com/cakedev0/talk-python-mauvais-pattern-en-data)

---

## Quel est le point commun de ces 3 fonctions?

```python
def merge_dicts(list_of_dicts: list[dict]) -> dict:
    return reduce(lambda d1, d2: {**d1, **d2}, list_of_dicts)
```
<!-- .element: class="fragment" -->

```python
def compute_n_uniques_by_user(df: pd.DataFrame) -> dict:
    n_uniques = {}
    for user in df['user'].unique():
        user_activity = df[df['user'] == user]
        n_uniques[user] = user_activity['item'].nunique()
    return n_uniques
```
<!-- .element: class="fragment" -->

```python
def compute_avg_by_group(group_idx: np.ndarray[int], x: np.ndarray) -> np.ndarray:
    n_groups = group_idx.max() + 1
    avg_by_group = np.empty(n_groups)
    for idx in range(n_groups):
        avg_by_group[idx] = np.mean(x[group_idx == idx])
    return avg_by_group
```
<!-- .element: class="fragment" -->


---

### Indice:

```python
def merge_dicts(list_of_dicts: list[dict]) -> dict:
    # n = sum(len(d) for d in list_of_dicts)
    return reduce(lambda d1, d2: {**d1, **d2}, list_of_dicts)
```

```python
def compute_n_uniques_by_user(df: pd.DataFrame) -> dict:
    # n = len(df)
    n_uniques = {}
    for user in df['user'].unique():
        user_activity = df[df['user'] == user]
        n_uniques[user] = user_activity['item'].nunique()
    return n_uniques
```

```python
def compute_avg_by_group(group_idx: np.ndarray[int], x: np.ndarray) -> np.ndarray:
    # n = x.size
    n_groups = group_idx.max() + 1
    avg_by_group = np.empty(n_groups)
    for idx in range(n_groups):
        avg_by_group[idx] = np.mean(x[group_idx == idx])
    return avg_by_group
```

---

## Réponse: L'inefficacité algorithmique!

Ces 3 algos peuvent être en $ O(n^2) $

---

## Un pattern pas si rare:

- Dictionnaires : vu tel quel dans script qui ne finissais jamais
- Pandas : croisé plusieurs fois chez des data scientists<!-- .element: class="fragment" -->
- Numpy : vu dans scikit-learn !<!-- .element: class="fragment" -->

-v-

## Pourquoi on écrit ça?

La complexité algorithmique est cachée par la syntaxe

```python
for user in df['user'].unique():
    user_activity = df[df['user'] == user]  # ← Process TOUTES les lignes!
    ...
```
<!-- .element: class="fragment" -->

<div class="fragment">
Perçu comme $ O(n) $ alors que c'est $ O(n^2) $
</div>


-v-

## Comment éviter ça?

1. Comprendre Python et ses librairies
2. Connaître les bons patterns<!-- .element: class="fragment" -->

---

## Solution dictionnaires

```python
merged_dict = {}
for d in list_of_dicts:
    merged_dict.update(d)  # O(len(d))
```

-v-

## Solution pandas

```python
n_uniques_by_user = df.groupby('user')['item'].nunique()
```

**Si c'est plus compliqué?**
<!-- .element: class="fragment" -->

```python
df.groupby('user').apply(some_complicated_function, include_groups=False)
```
<!-- .element: class="fragment" -->

<div class='fragment'> Plus lent (overhead par groupe) mais $ O(n) $ !</div>

-v-

## Solution numpy

```python
n_by_group = np.bincount(group_idx, minlength=n_groups)
sum_by_group = np.bincount(group_idx, weights=x, minlength=n_groups)

avg_by_group = sum_by_group / n_by_group
```

**Autre opération: minimum**
<!-- .element: class="fragment" -->

```python
min_by_group = np.full(n_groups, np.inf)
np.minimum.at(min_by_group, group_idx, x)
```
<!-- .element: class="fragment" -->

Les [ufuncs de numpy](https://numpy.org/doc/stable/reference/ufuncs.html#methods): `.at`, `.reduce`, `.accumulate`
<!-- .element: class="fragment" -->

---

# Merci !

---

# Numpy: pour aller plus loin

---

## 1. IDs non-denses

```python
x = np.random.rand(n)
group_id = np.random.choice(np.random.choice(10**18, n_groups), n)
```

```python
unique_ids, group_idx = np.unique(group_id, return_inverse=True)
# O(n log n), mais 10-100x plus lent que bincount en pratique
n_groups = unique_ids.size

sum_by_group = np.bincount(group_idx, weights=x)
```
<!-- .element: class="fragment" -->

---

## 2. Si les ufuncs ne suffisent pas

Exemple: médiane par groupe

#### Trick avec les matrices sparses<!-- .element: class="fragment" -->

```python
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
<!-- .element: class="fragment" -->


#### Pour la vitesse: numba<!-- .element: class="fragment" -->

```python
from numba import njit

median_per_row_numba = njit(median_per_row)

# ou Cython dans scikit-learn
```
<!-- .element: class="fragment" -->
