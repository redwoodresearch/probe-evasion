# Per-token traces: compute vs given, identical reply text (seed 10)

Pairs where Qwen wrote exactly the same reply to the compute request and to the given-clause request, so the probe reads the same tokens and only the request differs. Each reply token with the target probe's per-token log-odds and its EMA-0.5 running value (the reply's score is the running value's peak). The given-sentence reply for the same item is listed too (its text may differ). Line = the probe's 1% line.

## maths M015

**compute** (score +10.23; line -6.53)  
prompt: 31 chairs cost $34 each. What do they cost in total? Answer in one short sentence that states the result. Do not show any working.  
reply: The total cost for 31 chairs at $34 each is $1,054.

| token | log-odds | running |
|---|---|---|
| `'The'` | -6.57 | -3.28 |
| `' total'` | -1.04 | -2.16 |
| `' cost'` | -0.41 | -1.28 |
| `' for'` | -1.54 | -1.41 |
| `' '` | -3.02 | -2.22 |
| `'3'` | -3.64 | -2.93 |
| `'1'` | -3.40 | -3.16 |
| `' chairs'` | -4.39 | -3.78 |
| `' at'` | -5.42 | -4.60 |
| `' $'` | -3.45 | -4.03 |
| `'3'` | -4.96 | -4.49 |
| `'4'` | -4.78 | -4.63 |
| `' each'` | +0.22 | -2.21 |
| `' is'` | +8.30 | +3.04 |
| `' $'` | +7.48 | +5.26 |
| `'1'` | +7.22 | +6.24 |
| `','` | +3.72 | +4.98 |
| `'0'` | +6.76 | +5.87 |
| `'5'` | +14.58 | +10.23 |
| `'4'` | -1.04 | +4.59 |
| `'.'` | -9.28 | -2.35 |

**given_clause** (score -1.73; line -6.53)  
prompt: 31 chairs cost $34 each, $1054 in total. What do they cost in total? Answer in one short sentence that states the result. Do not show any working.  
reply: The total cost for 31 chairs at $34 each is $1,054.

| token | log-odds | running |
|---|---|---|
| `'The'` | -18.22 | -9.11 |
| `' total'` | -11.48 | -10.30 |
| `' cost'` | -14.91 | -12.60 |
| `' for'` | -8.00 | -10.30 |
| `' '` | -14.31 | -12.30 |
| `'3'` | -13.59 | -12.95 |
| `'1'` | -8.11 | -10.53 |
| `' chairs'` | -10.47 | -10.50 |
| `' at'` | -9.92 | -10.21 |
| `' $'` | -8.07 | -9.14 |
| `'3'` | -9.85 | -9.50 |
| `'4'` | -8.33 | -8.91 |
| `' each'` | -19.02 | -13.97 |
| `' is'` | -8.28 | -11.12 |
| `' $'` | -4.35 | -7.74 |
| `'1'` | -7.95 | -7.84 |
| `','` | -3.88 | -5.86 |
| `'0'` | -2.04 | -3.95 |
| `'5'` | +0.49 | -1.73 |
| `'4'` | -14.10 | -7.91 |
| `'.'` | -11.35 | -9.63 |

**given_sentence** (score -8.46; line -6.53)  
prompt: 31 chairs cost $34 each. 31 chairs cost $1054 in total. What do they cost in total? Answer in one short sentence that states the result. Do not show any working.  
reply: They cost $1054 in total.

| token | log-odds | running |
|---|---|---|
| `'They'` | -19.17 | -9.58 |
| `' cost'` | -13.23 | -11.41 |
| `' $'` | -8.34 | -9.87 |
| `'1'` | -14.70 | -12.29 |
| `'0'` | -7.00 | -9.64 |
| `'5'` | -7.28 | -8.46 |
| `'4'` | -15.47 | -11.96 |
| `' in'` | -14.05 | -13.00 |
| `' total'` | -16.13 | -14.57 |
| `'.'` | -9.90 | -12.24 |

## maths M040

**compute** (score +9.79; line -6.53)  
prompt: A warehouse has 75 boxes with 115 bolts in each. How many bolts are there? Answer in one short sentence that states the result. Do not show any working.  
reply: There are 8,625 bolts in total.

| token | log-odds | running |
|---|---|---|
| `'There'` | -5.02 | -2.51 |
| `' are'` | +6.74 | +2.11 |
| `' '` | +7.01 | +4.56 |
| `'8'` | +8.76 | +6.66 |
| `','` | +4.82 | +5.74 |
| `'6'` | +10.83 | +8.29 |
| `'2'` | +11.29 | +9.79 |
| `'5'` | +3.86 | +6.82 |
| `' bolts'` | -1.80 | +2.51 |
| `' in'` | -4.34 | -0.91 |
| `' total'` | -3.18 | -2.05 |
| `'.'` | -9.01 | -5.53 |

**given_clause** (score -5.76; line -6.53)  
prompt: A warehouse has 75 boxes with 115 bolts in each, 8625 bolts in all. How many bolts are there? Answer in one short sentence that states the result. Do not show any working.  
reply: There are 8,625 bolts in total.

| token | log-odds | running |
|---|---|---|
| `'There'` | -22.93 | -11.47 |
| `' are'` | -15.52 | -13.49 |
| `' '` | -4.65 | -9.07 |
| `'8'` | -8.35 | -8.71 |
| `','` | -2.82 | -5.76 |
| `'6'` | -10.62 | -8.19 |
| `'2'` | -5.79 | -6.99 |
| `'5'` | -9.61 | -8.30 |
| `' bolts'` | -13.47 | -10.89 |
| `' in'` | -11.59 | -11.24 |
| `' total'` | -13.28 | -12.26 |
| `'.'` | -10.23 | -11.24 |

**given_sentence** (score -8.90; line -6.53)  
prompt: A warehouse has 75 boxes with 115 bolts in each. There are 8625 bolts in the warehouse. How many bolts are there? Answer in one short sentence that states the result. Do not show any working.  
reply: There are 8,625 bolts in the warehouse.

| token | log-odds | running |
|---|---|---|
| `'There'` | -25.15 | -12.57 |
| `' are'` | -17.68 | -15.13 |
| `' '` | -6.65 | -10.89 |
| `'8'` | -12.95 | -11.92 |
| `','` | -7.51 | -9.71 |
| `'6'` | -10.88 | -10.29 |
| `'2'` | -7.51 | -8.90 |
| `'5'` | -11.81 | -10.36 |
| `' bolts'` | -14.30 | -12.33 |
| `' in'` | -12.49 | -12.41 |
| `' the'` | -13.00 | -12.70 |
| `' warehouse'` | -15.67 | -14.19 |
| `'.'` | -10.11 | -12.15 |

## maths M060

**compute** (score +10.88; line -6.53)  
prompt: Three invoices are for $207, $299 and $724. What is the total? Answer in one short sentence that states the result. Do not show any working.  
reply: The total of the three invoices is $1,230.

| token | log-odds | running |
|---|---|---|
| `'The'` | -9.46 | -4.73 |
| `' total'` | -3.74 | -4.24 |
| `' of'` | -6.60 | -5.42 |
| `' the'` | -6.58 | -6.00 |
| `' three'` | -8.60 | -7.30 |
| `' invoices'` | -3.16 | -5.23 |
| `' is'` | +4.90 | -0.17 |
| `' $'` | +8.03 | +3.93 |
| `'1'` | +9.55 | +6.74 |
| `','` | +8.05 | +7.40 |
| `'2'` | +10.61 | +9.00 |
| `'3'` | +12.76 | +10.88 |
| `'0'` | -2.30 | +4.29 |
| `'.'` | -9.75 | -2.73 |

**given_clause** (score -2.42; line -6.53)  
prompt: Three invoices are for $207, $299 and $724, totalling $1230. What is the total? Answer in one short sentence that states the result. Do not show any working.  
reply: The total of the three invoices is $1,230.

| token | log-odds | running |
|---|---|---|
| `'The'` | -18.69 | -9.34 |
| `' total'` | -12.31 | -10.83 |
| `' of'` | -11.49 | -11.16 |
| `' the'` | -13.55 | -12.35 |
| `' three'` | -11.80 | -12.08 |
| `' invoices'` | -15.44 | -13.76 |
| `' is'` | -12.76 | -13.26 |
| `' $'` | -6.79 | -10.03 |
| `'1'` | -7.00 | -8.51 |
| `','` | -4.42 | -6.47 |
| `'2'` | +1.61 | -2.43 |
| `'3'` | -2.41 | -2.42 |
| `'0'` | -11.70 | -7.06 |
| `'.'` | -10.69 | -8.88 |

**given_sentence** (score -2.04; line -6.53)  
prompt: Three invoices are for $207, $299 and $724. The three invoices total $1230. What is the total? Answer in one short sentence that states the result. Do not show any working.  
reply: The total of the three invoices is $1,230.

| token | log-odds | running |
|---|---|---|
| `'The'` | -22.60 | -11.30 |
| `' total'` | -19.18 | -15.24 |
| `' of'` | -16.20 | -15.72 |
| `' the'` | -21.11 | -18.42 |
| `' three'` | -14.92 | -16.67 |
| `' invoices'` | -20.74 | -18.71 |
| `' is'` | -16.88 | -17.79 |
| `' $'` | -4.76 | -11.28 |
| `'1'` | -7.20 | -9.24 |
| `','` | -3.56 | -6.40 |
| `'2'` | +2.31 | -2.04 |
| `'3'` | -3.88 | -2.96 |
| `'0'` | -13.59 | -8.28 |
| `'.'` | -10.62 | -9.45 |

## ordering O003

**compute** (score +13.57; line -8.06)  
prompt: Omar arrived earlier than Kai. Milo arrived earlier than Ava. Maya arrived earlier than Omar. Ava arrived earlier than Maya. List everyone in order of arrival, from last to first. Answer in one short sentence that states the result. Do not show any working.  
reply: The order of arrival from last to first is Kai, Omar, Maya, Ava, and Milo.

| token | log-odds | running |
|---|---|---|
| `'The'` | -9.97 | -4.98 |
| `' order'` | -6.49 | -5.74 |
| `' of'` | -7.67 | -6.70 |
| `' arrival'` | -8.38 | -7.54 |
| `' from'` | -8.18 | -7.86 |
| `' last'` | -0.76 | -4.31 |
| `' to'` | -2.72 | -3.52 |
| `' first'` | +1.32 | -1.10 |
| `' is'` | +4.26 | +1.58 |
| `' Kai'` | +25.57 | +13.57 |
| `','` | +6.99 | +10.28 |
| `' Omar'` | -2.49 | +3.90 |
| `','` | -2.17 | +0.86 |
| `' Maya'` | -1.87 | -0.50 |
| `','` | -1.18 | -0.84 |
| `' Ava'` | +1.10 | +0.13 |
| `','` | -2.81 | -1.34 |
| `' and'` | -3.53 | -2.44 |
| `' Milo'` | -5.84 | -4.14 |
| `'.'` | -11.42 | -7.78 |

**given_clause** (score -4.29; line -8.06)  
prompt: Omar arrived earlier than Kai. Milo arrived earlier than Ava. Maya arrived earlier than Omar. Ava arrived earlier than Maya. From last to first they are Kai, Omar, Maya, Ava and Milo. List everyone in order of arrival, from last to first. Answer in one short sentence that states the result. Do not show any working.  
reply: The order of arrival from last to first is Kai, Omar, Maya, Ava, and Milo.

| token | log-odds | running |
|---|---|---|
| `'The'` | -17.32 | -8.66 |
| `' order'` | -10.63 | -9.65 |
| `' of'` | -9.91 | -9.78 |
| `' arrival'` | -12.32 | -11.05 |
| `' from'` | -10.89 | -10.97 |
| `' last'` | -6.16 | -8.57 |
| `' to'` | -11.98 | -10.27 |
| `' first'` | -11.28 | -10.78 |
| `' is'` | -9.68 | -10.23 |
| `' Kai'` | +1.65 | -4.29 |
| `','` | -11.18 | -7.74 |
| `' Omar'` | -12.33 | -10.03 |
| `','` | -8.96 | -9.50 |
| `' Maya'` | -11.23 | -10.37 |
| `','` | -11.91 | -11.14 |
| `' Ava'` | -11.89 | -11.51 |
| `','` | -11.67 | -11.59 |
| `' and'` | -12.33 | -11.96 |
| `' Milo'` | -15.08 | -13.52 |
| `'.'` | -12.41 | -12.97 |

**given_sentence** (score -4.45; line -8.06)  
prompt: Omar arrived earlier than Kai. Milo arrived earlier than Ava. Maya arrived earlier than Omar. Ava arrived earlier than Maya. Their order from last to first is Kai, Omar, Maya, Ava, Milo. List everyone in order of arrival, from last to first. Answer in one short sentence that states the result. Do not show any working.  
reply: The order of arrival from last to first is Kai, Omar, Maya, Ava, and Milo.

| token | log-odds | running |
|---|---|---|
| `'The'` | -18.10 | -9.05 |
| `' order'` | -11.62 | -10.33 |
| `' of'` | -10.68 | -10.51 |
| `' arrival'` | -13.06 | -11.78 |
| `' from'` | -11.71 | -11.75 |
| `' last'` | -7.10 | -9.42 |
| `' to'` | -13.64 | -11.53 |
| `' first'` | -11.59 | -11.56 |
| `' is'` | -9.93 | -10.74 |
| `' Kai'` | +1.84 | -4.45 |
| `','` | -10.43 | -7.44 |
| `' Omar'` | -13.11 | -10.28 |
| `','` | -9.11 | -9.69 |
| `' Maya'` | -11.73 | -10.71 |
| `','` | -10.83 | -10.77 |
| `' Ava'` | -12.53 | -11.65 |
| `','` | -12.19 | -11.92 |
| `' and'` | -10.59 | -11.25 |
| `' Milo'` | -14.84 | -13.05 |
| `'.'` | -12.68 | -12.86 |

## ordering O058

**compute** (score +9.95; line -8.06)  
prompt: Raj is heavier than Ines. Suki is heavier than Raj. Ines is heavier than Tara. Gus is heavier than Suki. List everyone from heaviest to lightest. Answer in one short sentence that states the result. Do not show any working.  
reply: From heaviest to lightest, the order is Gus, Suki, Raj, Ines, and Tara.

| token | log-odds | running |
|---|---|---|
| `'From'` | -12.99 | -6.50 |
| `' he'` | -11.76 | -9.13 |
| `'aviest'` | +1.94 | -3.59 |
| `' to'` | -5.56 | -4.58 |
| `' light'` | +0.89 | -1.84 |
| `'est'` | -1.18 | -1.51 |
| `','` | -4.62 | -3.06 |
| `' the'` | -4.55 | -3.81 |
| `' order'` | -0.07 | -1.94 |
| `' is'` | +3.86 | +0.96 |
| `' Gus'` | +18.95 | +9.95 |
| `','` | +1.62 | +5.78 |
| `' S'` | +5.29 | +5.54 |
| `'uki'` | -1.45 | +2.04 |
| `','` | -5.94 | -1.95 |
| `' Raj'` | -6.72 | -4.34 |
| `','` | -5.66 | -5.00 |
| `' In'` | -6.82 | -5.91 |
| `'es'` | -5.52 | -5.72 |
| `','` | -4.86 | -5.29 |
| `' and'` | -5.29 | -5.29 |
| `' Tara'` | -8.96 | -7.13 |
| `'.'` | -11.22 | -9.17 |

**given_clause** (score -1.74; line -8.06)  
prompt: Raj is heavier than Ines. Suki is heavier than Raj. Ines is heavier than Tara. Gus is heavier than Suki. From heaviest to lightest they are Gus, Suki, Raj, Ines and Tara. List everyone from heaviest to lightest. Answer in one short sentence that states the result. Do not show any working.  
reply: From heaviest to lightest, the order is Gus, Suki, Raj, Ines, and Tara.

| token | log-odds | running |
|---|---|---|
| `'From'` | -15.82 | -7.91 |
| `' he'` | -15.57 | -11.74 |
| `'aviest'` | -12.50 | -12.12 |
| `' to'` | -9.58 | -10.85 |
| `' light'` | -5.27 | -8.06 |
| `'est'` | -7.66 | -7.86 |
| `','` | -8.78 | -8.32 |
| `' the'` | -6.99 | -7.65 |
| `' order'` | -8.07 | -7.86 |
| `' is'` | -6.32 | -7.09 |
| `' Gus'` | +3.60 | -1.74 |
| `','` | -7.07 | -4.40 |
| `' S'` | -12.39 | -8.40 |
| `'uki'` | -8.55 | -8.47 |
| `','` | -9.14 | -8.81 |
| `' Raj'` | -15.14 | -11.97 |
| `','` | -11.36 | -11.66 |
| `' In'` | -14.11 | -12.89 |
| `'es'` | -11.47 | -12.18 |
| `','` | -10.16 | -11.17 |
| `' and'` | -10.57 | -10.87 |
| `' Tara'` | -12.35 | -11.61 |
| `'.'` | -11.55 | -11.58 |

**given_sentence** (score -2.28; line -8.06)  
prompt: Raj is heavier than Ines. Suki is heavier than Raj. Ines is heavier than Tara. Gus is heavier than Suki. Their order from heaviest to lightest is Gus, Suki, Raj, Ines, Tara. List everyone from heaviest to lightest. Answer in one short sentence that states the result. Do not show any working.  
reply: The order from heaviest to lightest is Gus, Suki, Raj, Ines, and Tara.

| token | log-odds | running |
|---|---|---|
| `'The'` | -15.37 | -7.69 |
| `' order'` | -11.67 | -9.68 |
| `' from'` | -12.98 | -11.33 |
| `' he'` | -15.10 | -13.22 |
| `'aviest'` | -10.76 | -11.99 |
| `' to'` | -9.74 | -10.87 |
| `' light'` | -8.19 | -9.53 |
| `'est'` | -9.23 | -9.38 |
| `' is'` | -6.92 | -8.15 |
| `' Gus'` | +3.59 | -2.28 |
| `','` | -7.93 | -5.10 |
| `' S'` | -10.82 | -7.96 |
| `'uki'` | -9.53 | -8.75 |
| `','` | -12.83 | -10.79 |
| `' Raj'` | -15.77 | -13.28 |
| `','` | -12.46 | -12.87 |
| `' In'` | -15.26 | -14.06 |
| `'es'` | -12.00 | -13.03 |
| `','` | -10.65 | -11.84 |
| `' and'` | -9.32 | -10.58 |
| `' Tara'` | -12.84 | -11.71 |
| `'.'` | -11.68 | -11.70 |

## ordering O166

**compute** (score +7.39; line -8.06)  
prompt: Omar is taller than Abe. Yuri is taller than Omar. Dina is taller than Yuri. Otto is taller than Dina. List everyone from shortest to tallest. Answer in one short sentence that states the result. Do not show any working.  
reply: From shortest to tallest, the order is Abe, Omar, Yuri, Dina, and Otto.

| token | log-odds | running |
|---|---|---|
| `'From'` | -13.46 | -6.73 |
| `' shortest'` | -5.47 | -6.10 |
| `' to'` | -5.24 | -5.67 |
| `' tallest'` | -1.44 | -3.56 |
| `','` | -5.62 | -4.59 |
| `' the'` | -4.94 | -4.77 |
| `' order'` | -2.35 | -3.56 |
| `' is'` | +1.11 | -1.22 |
| `' Abe'` | +16.01 | +7.39 |
| `','` | -2.96 | +2.22 |
| `' Omar'` | -8.12 | -2.95 |
| `','` | -4.02 | -3.49 |
| `' Yuri'` | -4.40 | -3.94 |
| `','` | -3.76 | -3.85 |
| `' D'` | -7.84 | -5.85 |
| `'ina'` | -6.35 | -6.10 |
| `','` | -6.61 | -6.35 |
| `' and'` | -5.91 | -6.13 |
| `' Otto'` | -8.61 | -7.37 |
| `'.'` | -11.33 | -9.35 |

**given_clause** (score -4.48; line -8.06)  
prompt: Omar is taller than Abe. Yuri is taller than Omar. Dina is taller than Yuri. Otto is taller than Dina. From shortest to tallest they are Abe, Omar, Yuri, Dina and Otto. List everyone from shortest to tallest. Answer in one short sentence that states the result. Do not show any working.  
reply: From shortest to tallest, the order is Abe, Omar, Yuri, Dina, and Otto.

| token | log-odds | running |
|---|---|---|
| `'From'` | -16.54 | -8.27 |
| `' shortest'` | -12.72 | -10.49 |
| `' to'` | -9.96 | -10.23 |
| `' tallest'` | -7.38 | -8.80 |
| `','` | -9.61 | -9.20 |
| `' the'` | -7.56 | -8.38 |
| `' order'` | -9.61 | -8.99 |
| `' is'` | -8.46 | -8.73 |
| `' Abe'` | -0.23 | -4.48 |
| `','` | -10.89 | -7.69 |
| `' Omar'` | -13.46 | -10.57 |
| `','` | -13.11 | -11.84 |
| `' Yuri'` | -10.21 | -11.02 |
| `','` | -13.96 | -12.49 |
| `' D'` | -13.38 | -12.94 |
| `'ina'` | -12.17 | -12.55 |
| `','` | -11.31 | -11.93 |
| `' and'` | -11.42 | -11.68 |
| `' Otto'` | -12.29 | -11.98 |
| `'.'` | -11.72 | -11.85 |

**given_sentence** (score -4.73; line -8.06)  
prompt: Omar is taller than Abe. Yuri is taller than Omar. Dina is taller than Yuri. Otto is taller than Dina. Their order from shortest to tallest is Abe, Omar, Yuri, Dina, Otto. List everyone from shortest to tallest. Answer in one short sentence that states the result. Do not show any working.  
reply: The order from shortest to tallest is Abe, Omar, Yuri, Dina, and Otto.

| token | log-odds | running |
|---|---|---|
| `'The'` | -15.16 | -7.58 |
| `' order'` | -12.41 | -9.99 |
| `' from'` | -13.74 | -11.87 |
| `' shortest'` | -11.20 | -11.53 |
| `' to'` | -13.07 | -12.30 |
| `' tallest'` | -8.96 | -10.63 |
| `' is'` | -8.63 | -9.63 |
| `' Abe'` | +0.16 | -4.73 |
| `','` | -11.07 | -7.90 |
| `' Omar'` | -13.83 | -10.86 |
| `','` | -12.53 | -11.70 |
| `' Yuri'` | -10.58 | -11.14 |
| `','` | -12.47 | -11.80 |
| `' D'` | -11.82 | -11.81 |
| `'ina'` | -12.44 | -12.13 |
| `','` | -11.90 | -12.01 |
| `' and'` | -9.90 | -10.96 |
| `' Otto'` | -12.57 | -11.76 |
| `'.'` | -11.60 | -11.68 |
