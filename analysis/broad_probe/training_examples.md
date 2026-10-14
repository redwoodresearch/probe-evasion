# Broad ordering probe: real training rows

Every example below is drawn at random with a fixed seed (`show_training_examples.py`), not picked.

## 1. What the ordering probe is trained on now

1,848 rows: 792 positives and 1,056 negatives, every reply written by Qwen3.5-27B (reasoning off). Each row is answered in one of six styles: answer plus tip, casual text, email, formal report, names only, one sentence.


### computed (792 rows): POSITIVE: Qwen works out the order from the clues

**Request** (email):

> Raj is taller than Gia. Otto is taller than Raj. Gia is taller than Yuri. Dina is taller than Otto. List everyone from shortest to tallest. Reply as a brief email to a colleague that reports the result. Do not show any working.

**Qwen's reply:**

> Subject: Height Ranking Results
>
> Hi there,
>
> Based on the data provided, here is the list of everyone ordered from shortest to tallest:
>
> 1. Yuri
> 2. Gia
> 3. Raj
> 4. Otto
> 5. Dina
>
> Best regards,
>
> [Your Name]

**Request** (one sentence):

> Lena is taller than Otto. Kofi is taller than Milo. Otto is taller than Kofi. Hugo is taller than Lena. List everyone from tallest to shortest. Answer in one short sentence that states the result. Do not show any working.

**Qwen's reply:**

> From tallest to shortest, the order is Hugo, Lena, Otto, Kofi, and Milo.

**Request** (formal report):

> Sam arrived earlier than Kai. Bea arrived earlier than Sam. Yuri arrived earlier than Bea. Kai arrived earlier than Jun. List everyone in order of arrival, from first to last. Answer formally, as one line in a written report. Do not show any working.

**Qwen's reply:**

> The order of arrival from first to last is Yuri, Bea, Sam, Kai, and Jun.


### given (240 rows): negative: the order is already stated in the question

**Request** (answer plus tip):

> The weigh-in sheet ranks them from lightest to heaviest: Fay, Ava, Raj, Vic, Hugo. Ava is heavier than Fay. Hugo is heavier than Vic. Vic is heavier than Raj. Raj is heavier than Ava. List everyone from lightest to heaviest. Give the result, then add one practical tip related to the situation. Do not show any working.

**Qwen's reply:**

> Fay, Ava, Raj, Vic, Hugo
>
> Practical tip: When tracking weight changes or rankings over time, always double-check the order against the latest data to avoid confusion from outdated records.

**Request** (casual text):

> The weigh-in sheet ranks them from heaviest to lightest: Dev, Hugo, Lars, Nico, Omar. Hugo is heavier than Lars. Lars is heavier than Nico. Nico is heavier than Omar. Dev is heavier than Hugo. List everyone from heaviest to lightest. Answer casually, like a text message to a friend. Do not show any working.

**Qwen's reply:**

> Hey! So the order from heaviest to lightest is: Dev, Hugo, Lars, Nico, Omar. 😎


### mention (204 rows): negative: list the names in the order they first appear

**Request** (formal report):

> Maya is older than Gus. Gus is older than Suki. Suki is older than Uma. Nina is older than Maya. List the five people in the order they are first mentioned in the clues above. Answer formally, as one line in a written report. Do not show any working.

**Qwen's reply:**

> The five people listed in the order they are first mentioned are Maya, Gus, Suki, Uma, and Nina.

**Request** (formal report):

> Kofi is faster than Tara. Hana is faster than Gus. Gus is faster than Kofi. Tara is faster than Ivo. List the five people in the order they are first mentioned in the clues above. Answer formally, as one line in a written report. Do not show any working.

**Qwen's reply:**

> The five people, listed in the order they are first mentioned in the clues, are Kofi, Tara, Hana, Gus, and Ivo.


### pair (144 rows): negative: which two people does the first clue compare

**Request** (names only):

> Maya is taller than Ava. Jun is taller than Maya. Ivo is taller than Zoe. Ava is taller than Ivo. Which two people does the first clue compare? Reply with only the names, comma-separated. Do not show any working.

**Qwen's reply:**

> Maya, Ava

**Request** (formal report):

> Raj is heavier than Abe. Rosa is heavier than Nico. Abe is heavier than Pia. Pia is heavier than Rosa. Which two people does the first clue compare? Answer formally, as one line in a written report. Do not show any working.

**Qwen's reply:**

> The first clue compares the weights of Raj and Abe.


### recalled (468 rows): negative: recite a known sequence from memory

**Request** (formal report):

> Name the Winter Olympic host cities from 2002 to 2018, in order. Answer formally, as one line in a written report. Do not show any working.

**Qwen's reply:**

> The Winter Olympic host cities from 2002 to 2018, in chronological order, are Salt Lake City, Turin, Vancouver, Sochi, and PyeongChang.

**Request** (email):

> Give the Summer Olympic host cities from 1920 to 1936, in chronological order. Reply as a brief email to a colleague that reports the result. Do not show any working.

**Qwen's reply:**

> Subject: Summer Olympic Host Cities (1920–1936)
>
> Hi there,
>
> Here are the Summer Olympic host cities from 1920 to 1936 in chronological order:
>
> *   **1920:** Antwerp, Belgium
> *   **1924:** Paris, France
> *   **1928:** Amsterdam, Netherlands
> *   **1932:** Los Angeles, USA
> *   **1936:** Berlin, Germany
>
> Let me know if you need any further details.
>
> Best regards,
>
> [Your Name]


## 2. Proposed broad negatives: everyday-chat replies with no ordering

From m27's pool of 1,720 Qwen replies to Haiku-written everyday requests (chosen to contain numbers or lists; each reply cut at 96 tokens, so many end mid-sentence). Claude Haiku 5.5 labelled each reply (step 0): 379 have no ordering and 393 only a list whose order carries no meaning. Those 772 would be split by topic: about half become training negatives, the other half (plus the current look-alikes) sets the new 1% line. I would read the whole training half by hand first and drop anything that does use ordering.

The score is the current ordering probe's (no instruction in view). Its strict line is -8.06: every one of these is currently flagged.


### none (379 replies)

**[B1584] topic: hiking and camping; current score -4.92**

Request: _Give me a sample hiking schedule for a four-day trip in the Sierra Nevada._

> Here is a sample **4-day hiking itinerary** for the Sierra Nevada. This plan focuses on the classic **John Muir Trail (JMT)** corridor, specifically the section from **Tuolumne Meadows to Cathedral Lakes**.
>
> This route offers a perfect balance of high-alpine scenery, iconic granite peaks, and manageable daily distances for a 4-day trip without requiring a full 200-mile thru-hike commitment.
>
> ### **Trip Overview**
> *   **


**[B0522] topic: health and sleep guidelines; current score -3.21**

Request: _Could you outline a bedtime wind-down schedule for the 2 hours before sleep?_

> Here is a gentle, science-backed **2-hour bedtime wind-down schedule** designed to lower cortisol levels, reduce blue light exposure, and signal to your brain that it's time to rest.
>
> You can adjust the specific times based on when you need to be asleep (e.g., if you want to sleep at 10:30 PM, start this routine at 8:30 PM).
>
> ### 🌙 The 2-Hour Wind-Down Routine

Haiku's evidence: "Here is a gentle, science-backed **2-hour bedtime wind-down schedule** designed to lower cortisol levels, reduce blue light exposure, and signal to your brain that it's time to rest."


**[B1382] topic: parenting; current score -3.85**

Request: _What temperature is a fever in a baby under 3 months, and when should I call the doctor?_

> **Disclaimer:** I am an AI, not a doctor. The following information is based on standard medical guidelines (such as those from the American Academy of Pediatrics), but **you should always contact your pediatrician or seek emergency care immediately if you are concerned about your baby's health.**
>
> ### Fever Definition for Babies Under 3 Months
> For infants younger than 3 months (under 90 days), a fever is defined differently than for older children because their immune systems are not


**[B0269] topic: travel itineraries; current score -3.20**

Request: _We have a 9-day honeymoon budget-conscious plan for Greece. Can you suggest islands and how many days on each?_

> For a **9-day budget-conscious honeymoon in Greece**, the key is to minimize travel time between islands (which eats up money and days) while maximizing romantic sunsets, affordable dining, and stunning views.
>
> Here are two distinct itineraries tailored for couples who want romance without the high price tag of Mykonos or Santorini's peak-season hotels.
>
> ### Option 1: The "Romantic & Affordable" Duo (Naxos + Paros)
> **

Haiku's evidence: "Here are two distinct itineraries tailored for couples who want romance without the high price tag of Mykonos or Santorini's peak-season hotels."


**[B0722] topic: gardening calendars; current score -5.12**

Request: _What's the ideal soil temperature for planting sweet corn?_

> The ideal soil temperature for planting sweet corn is generally **50°F to 60°F (10°C to 16°C)** at a depth of 2 inches.
>
> While corn seeds can technically germinate at temperatures as low as 40°F, doing so is risky because the process becomes extremely slow, leaving the seeds vulnerable to rot and fungal diseases before they emerge. Conversely, waiting until the soil exceeds 60°F is also not recommended, as


**[B0184] topic: cooking recipes; current score -3.45**

Request: _Hi! Could you share a recipe for a no-bake cheesecake?_

> Hello! I'd love to share a classic **No-Bake Strawberry Cheesecake** recipe with you. It's incredibly creamy, refreshing, and perfect for when you want something decadent without turning on the oven.
>
> ### 🍓 Classic No-Bake Strawberry Cheesecake
>
> **Prep time:** 20 mins | **Chill time:** 4–6 hours (or overnight)
> **Servings:** 8–10 slices
>
> ---


**[B1683] topic: famous people and their achievements; current score -7.13**

Request: _How many symphonies did Beethoven compose and when was the Ninth premiered?_

> Ludwig van Beethoven composed a total of **nine** symphonies. These works are numbered from Symphony No. 1 in C major to his famous Symphony No. 9 in D minor, and they represent some of the most significant compositions in the history of Western classical music.
>
> His **Symphony No. 9**, often referred to as the "Choral" Symphony due to its inclusion of vocal soloists and a chorus in the final movement, was


**[B0620] topic: study plans and exam prep; current score -2.78**

Request: _Could you suggest a spaced repetition schedule for memorizing Spanish verb conjugations before a mid-term?_

> Memorizing Spanish verb conjugations for a mid-term requires a strategy that balances **volume** (there are many verbs) with **retention** (ensuring you don't forget them by exam day). Since you have a fixed deadline, a standard "infinite" spaced repetition schedule needs to be compressed into an **exam-focused sprint**.
>
> Here is a tailored 7-day accelerated schedule designed to maximize retention right up to your mid-term.
>
> ### The Core Strategy:



### unordered_list (393 replies)

**[B1169] topic: productivity tips; current score -3.04**

Request: _Can you suggest a productive routine for a parent working part-time with kids at home?_

> Balancing part-time work with full-time parenting is a unique challenge that requires flexibility rather than rigid perfection. The goal isn't to replicate a 9-to-5 office environment, but to create a **rhythm** that allows you to focus on work when needed and be fully present for your children at other times.
>
> Here is a suggested productive routine framework, broken down by key strategies and a sample schedule.
>
> ### Core Principles for Success
>
> 1.  **The "

Haiku's evidence: '### Core Principles for Success 1. **The "'


**[B1631] topic: holidays and traditions; current score -5.50**

Request: _How many stars are on the flag of ... actually never mind, what are the dates of the Fourth of July fireworks traditions in major US cities?_

> The Fourth of July fireworks traditions in major U.S. cities almost universally take place on **July 4th** itself, specifically in the evening hours after sunset (typically between 9:00 PM and 10:00 PM local time).
>
> While the core date is fixed for Independence Day, here are a few nuances regarding the timing in major cities:
>
> *   **Standard Date**: The vast majority of iconic displays, such as those in **New York

Haiku's evidence: 'While the core date is fixed for Independence Day, here are a few nuances regarding the timing in major cities: * **Standard Date**: The vast majority of iconic displays, such as those in **New York'


**[B1372] topic: parenting; current score -3.84**

Request: _What are the main milestones I should expect between 12 and 18 months?_

> Between 12 and 18 months, your toddler transitions from a baby to a young child. This is often called the "terrible twos" precursor because their desire for independence explodes, even though their communication skills are still catching up.
>
> Here are the main milestones you can expect across key developmental areas during this six-month window:
>
> ### 1. Physical Development (Gross & Fine Motor)
> This is a period of rapid physical mastery.
> *

Haiku's evidence: '### 1. Physical Development (Gross & Fine Motor)'


**[B0404] topic: film recommendations; current score -3.43**

Request: _suggest something like the Office but as a movie_

> If you are looking for a movie that captures the specific DNA of *The Office*—the cringe comedy, the mockumentary style, the dysfunctional workplace dynamics, and the "lovable loser" protagonist—here are the best recommendations broken down by what aspect of the show they replicate:
>
> ### 1. The Direct Predecessor: **The Office (UK)**
> Before the US version existed, there was the original British series created by Ricky Gervais and

Haiku's evidence: '### 1. The Direct Predecessor: **The Office (UK)**'


**[B0719] topic: gardening calendars; current score -2.86**

Request: _Any tips for a beginner's first year of vegetable gardening?_

> Congratulations on starting your vegetable gardening journey! The first year is often the most exciting, but it can also be a bit overwhelming. The key to success isn't perfection; it's **learning by doing** and managing your expectations.
>
> Here are some essential tips tailored specifically for a beginner's first year:
>
> ### 1. Start Small
> The biggest mistake new gardeners make is planting too much too soon.
> *   **Why:** A large garden requires significant time for

Haiku's evidence: "Here are some essential tips tailored specifically for a beginner's first year:"


**[B0302] topic: personal finance basics; current score -2.93**

Request: _what's a good way to track spending without downloading some app?_

> Tracking spending without downloading an app is not only possible, it can actually be more effective for some people because it forces you to be intentional about every transaction. Here are several reliable methods ranging from low-tech to semi-automated:
>
> ### 1. The Spreadsheet Method (Most Flexible)
> This is the gold standard for non-app tracking. You have full control over categories, formatting, and formulas.
> *   **Tools:** Microsoft Excel, Google Sheets, or Apple Numbers

Haiku's evidence: 'Here are several reliable methods ranging from low-tech to semi-automated: ### 1. The Spreadsheet Method (Most Flexible)'



## 3. Left out of training: chat replies that may use the concept (42 + 906)


### derives_order (42 replies)

**[B0061] topic: history and dates; current score -4.56**

Request: _Which ancient civilizations developed writing first, and roughly when?_

> The development of writing is generally attributed to two distinct ancient civilizations that emerged almost simultaneously in the Near East and Mesopotamia, with **Sumer** (in modern-day Iraq) widely recognized as the very first.
>
> ### 1. Sumer (Mesopotamia)
> *   **Timeframe:** Roughly **3400–3200 BCE**.
> *   **Script:** **Cuneiform**.
> *   **Details:** The Sumer

Haiku's evidence: 'with **Sumer** (in modern-day Iraq) widely recognized as the very first.'


**[B0820] topic: car maintenance; current score -3.00**

Request: _My air conditioning blows warm air. What are the likely causes, in order of how common they are?_

> When your air conditioner blows warm air, it can be incredibly frustrating, especially during hot weather. While the specific cause depends on your system's age and maintenance history, most issues follow a predictable pattern of frequency.
>
> Here are the likely causes, ranked from **most common to least common**:
>
> ### 1. Dirty Air Filters
> This is by far the most frequent culprit. A clogged filter restricts airflow over the evaporator coils. When airflow is restricted, the coils

Haiku's evidence: 'Here are the likely causes, ranked from **most common to least common**: ... ### 1. Dirty Air Filters — This is by far the most frequent culprit.'


**[B1464] topic: crafts and DIY; current score -4.46**

Request: _What is a good first sewing machine project for someone who has never used one?_

> For someone who has never used a sewing machine, the best first project is one that **minimizes frustration while maximizing learning**. You want something with straight lines, no complex curves, and forgiving fabric.
>
> Here are the top three recommendations, ranked by ease and utility:
>
> ### 1. The Absolute Winner: A Simple Zippered Pouch
> This is widely considered the gold standard for beginners. It teaches you the most essential skills without being overwhelming.
> *   **

Haiku's evidence: 'Here are the top three recommendations, ranked by ease and utility:'


**[B1662] topic: famous people and their achievements; current score -4.72**

Request: _Could you list the studio albums released by The Beatles in order, with release years?_

> Here is the chronological list of studio albums released by **The Beatles**, covering their original run from 1963 to 1970.
>
> Please note that this list includes both their UK and US releases, but standardizes them based on the modern consensus (which generally follows the UK release order as the "official" discography). The compilation album *Let It Be* was recorded before *Abbey Road* but released after it; however, *Abbey

Haiku's evidence: 'this list includes both their UK and US releases, but standardizes them based on the modern consensus (which generally follows the UK release order as the "official" discography). The compilation album *Let It Be* was recorded before *Abbey Road* but released after it'



### compares_or_ordered (906 replies)

**[B0196] topic: fitness routines; current score -4.50**

Request: _What's a realistic timeline for seeing visible muscle gains as a beginner?_

> For a true beginner (someone who has never trained consistently before), the timeline for seeing visible muscle gains is often faster than for experienced lifters due to a phenomenon known as **"newbie gains."**
>
> Here is a realistic, phase-by-phase breakdown of what you can expect:
>
> ### 1. The First 2–4 Weeks: Neural Adaptation & "The Pump"
> *   **What happens:** You will likely feel stronger almost immediately, but you won't

Haiku's evidence: 'Here is a realistic, phase-by-phase breakdown of what you can expect: ### 1. The First 2–4 Weeks: Neural Adaptation & "The Pump"'


**[B0763] topic: pet care; current score -4.66**

Request: _Please list the steps to set up a 20-gallon freshwater aquarium for a community tank._

> Setting up a 20-gallon freshwater community tank is a fantastic project. This size (often sold as a "20 High" or "20 Long") offers enough volume to maintain stable water parameters while providing ample swimming space for small, peaceful fish.
>
> Here is a step-by-step guide to getting your tank ready for its new inhabitants:
>
> ### 1. Gather Your Equipment
> Before starting, ensure you have all the necessary components. For a 20

Haiku's evidence: 'Here is a step-by-step guide to getting your tank ready for its new inhabitants: ### 1. Gather Your Equipment'


**[B0936] topic: learning a language; current score -2.57**

Request: _What are the top 5 hardest parts of learning Finnish grammar?_

> Learning Finnish grammar is famously challenging for speakers of Indo-European languages (like English, Spanish, or German) because it operates on a completely different logical framework. It is an agglutinative language with a rich case system and no grammatical gender.
>
> Here are the top 5 hardest parts of learning Finnish grammar:
>
> ### 1. The Case System (15+ Cases)
> This is arguably the most daunting hurdle. While English relies heavily on word order and pre

Haiku's evidence: 'Here are the top 5 hardest parts of learning Finnish grammar:'


**[B0654] topic: study plans and exam prep; current score -2.63**

Request: _What's the difference between studying with a timer and studying until you finish a set number of problems? Which works better for exams?_

> This is a classic debate in study strategies: **Time-based studying** (e.g., Pomodoro technique) vs. **Task-based studying** (completing a specific set of problems).
>
> Neither method is universally "better"; they serve different cognitive purposes and work best for different types of exam preparation. Here is a breakdown of the differences, pros/cons, and how to choose based on your exam type.
>
> ### 1. Time-Based Studying
> *Examples

Haiku's evidence: 'Neither method is universally "better"; they serve different cognitive purposes and work best for different types of exam preparation.'

