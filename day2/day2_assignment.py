import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set a consistent style for our plots
sns.set_style('darkgrid')
netflix_df = pd.read_csv('netflix_titles.csv')

"""
Q1. How has the distribution of content ratings changed over time?
"""
netflix_df["release_year"] = netflix_df["release_year"].astype(int)
print(netflix_df.dtypes)
rating_over_time = netflix_df.groupby(["year_added", 'rating']).size().unstack().fillna(0)
print(rating_over_time)
plt.figure(figsize=(14, 8))
rating_over_time.plot(kind='line', marker='o',figsize=(14, 8))
plt.title('content ratings changed over time')
plt.xlabel('year_added')
plt.ylabel('Number of ratings Added')
plt.legend(title='Content ratings')
plt.show()

# 1. Handle missing values in 'director' and 'cast'
# Since these are text fields and many are missing, we'll fill them with 'Unknown'.
netflix_df['director'] = netflix_df['director'].fillna('Unknown')
netflix_df['cast'] = netflix_df['cast'].fillna('Unknown')

# 2. Handle missing 'country'
# We'll fill with the mode, which is the most common country.
mode_country = netflix_df['country'].mode()[0]
netflix_df['country'] = netflix_df['country'].fillna(mode_country)

netflix_df['date_added'] = pd.to_datetime(netflix_df['date_added'], format="mixed")
netflix_df.head()

# 5. Create new features for year and month added
netflix_df['year_added'] = netflix_df['date_added'].dt.year
netflix_df['month_added'] = netflix_df['date_added'].dt.month

# Verify our cleaning and transformation
print("Missing values after cleaning:")
print(netflix_df.isnull().sum())
print("\nData types after transformation:")
print(netflix_df.dtypes)

# Create the 'age_on_netflix' feature
netflix_df['age_on_netflix'] = netflix_df['year_added'] - netflix_df['release_year']

# Filter out any potential errors where added_year is before release_year
content_age = netflix_df[netflix_df['age_on_netflix'] >= 0]

"""
Insights: 
Insights: ⭐ 1. TV-MA dominates after 2016
TV-MA shows the largest spike, peaking around 2019–2020 with nearly 800 titles added.
This suggests a strong shift toward mature-audience content, likely driven by streaming platforms investing heavily in darker, edgier shows.
⭐ 2. TV-14 is the second-biggest growth category
It rises sharply from 2016 and peaks around 2019, indicating high output aimed at teens and general audiences.
⭐ 3. Family and kids ratings grow but more modestly
Categories like TV-PG, TV-Y, and TV-Y7 show noticeable growth but stay well below the adult categories.
Their peaks in 2018–2019 indicate steady but not explosive investment in family content.
⭐ 4. Movie ratings (PG, PG-13, R, G) stay much lower
They grow slightly but remain far below TV ratings, confirming a shift in focus toward TV series over movies on streaming platforms.
⭐ 5. Sharp drop after 2020
All categories decline sharply in 2021, likely due to:
Pandemic disruptions in production
Content release delays
Shifts in streaming strategy
"""

"""
Q2. Is there a relationship between content age and its type (Movie vs. TV Show)?
"""
print(content_age.tail())
sns.catplot(x='type', y='age_on_netflix', hue='type', data=content_age, kind='bar', height=10, aspect=1.5)
plt.title('relationship between content age and its type')
plt.ylabel('content added wrt type')
plt.show()

"""Insights:
Insight: Here we can clearly see that movies have more age than tv shows"""

"""
Q3. Can we identify any trends in content production based on the release year vs. the year added to Netflix?
"""
plt.figure(figsize=(12,6))
sns.scatterplot(data=netflix_df, x='release_year', y='year_added')
plt.title("Release Year vs Year Added to Netflix")
plt.xlabel("Release Year")
plt.ylabel("Added to Netflix")
plt.show()

"""Insights: Netflix is adding older catalog content recently"""

"""
Q4. What are the most common word pairs or phrases in content descriptions?
"""
from collections import Counter

pairs = Counter()

for desc in netflix_df['description'].astype(str):
    words = desc.lower().split()
    for i in range(len(words)-1):
        pair = words[i] + " " + words[i+1]
        pairs[pair] += 1

# Convert top 10 to DataFrame
top_pairs = pd.DataFrame(pairs.most_common(10), columns=['pair', 'count'])

plt.figure(figsize=(10,5))
plt.barh(top_pairs['pair'], top_pairs['count'])
plt.xlabel("Frequency")
plt.ylabel("Word Pair (Bigram)")
plt.title("Top 10 Most Common Word Pairs in Content Descriptions")
plt.gca().invert_yaxis()   # highest appears on top
plt.show()

"""Insights: most common word pair: 'in a' """

"""
Q5. Who are the top directors on Netflix?
"""
director_df = netflix_df[netflix_df['director'] != "Unknown"]
director_df.head()
top_directors_counts = director_df['director'].value_counts().reset_index()
print(top_directors_counts)
top_directors_counts_plot = top_directors_counts.head(15)

plt.figure(figsize=(12, 8))
sns.barplot(top_directors_counts_plot, y='director', x='count',palette='mako',hue='director')
plt.show()

"""
Insights:

Raúl Campos, Jan Suter has directed most number of movies
"""