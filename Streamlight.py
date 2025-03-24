import streamlit as st
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import re
import nltk

nltk.download('punkt') 

import os
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Preprocessing functions
ps = PorterStemmer()
stopwords = set(["in", "for", "the", "of", "and", "a", "an", "to", "on", "by"])


def tokenize_and_stem(keyword):
    keyword = keyword.lower()
    keyword = re.sub(r'[^\w\s]', '', keyword)
    tokens = word_tokenize(keyword)
    tokens = [t for t in tokens if t not in stopwords]
    stems = [ps.stem(t) for t in tokens]
    return stems


def preprocess_keywords(keywords):
    processed = []
    for kw in keywords:
        stems = tokenize_and_stem(kw)
        processed.append({'original': kw, 'stems': set(stems), 'stems_list': stems})
    return processed


def cluster_keywords(data, cluster_key, level=1, max_level=3):
    exact = []
    extended = []
    for kw in data:
        if cluster_key.issubset(kw['stems']):
            if len(kw['stems']) == len(cluster_key):
                exact.append(kw['original'])
            elif len(kw['stems']) > len(cluster_key):
                extended.append(kw)
    result = {'cluster_key': cluster_key, 'exact': exact}
    if level == max_level or not extended:
        result['extended'] = [kw['original'] for kw in extended]
        return result
    candidate_counter = Counter()
    for kw in extended:
        additional = kw['stems'] - cluster_key
        candidate_counter.update(additional)
    branches = {}
    for candidate, _ in candidate_counter.most_common():
        candidate_cluster_key = set(cluster_key) | {candidate}
        candidate_keywords = [kw for kw in extended if candidate_cluster_key.issubset(kw['stems'])]
        if candidate_keywords:
            branch_result = cluster_keywords(candidate_keywords, candidate_cluster_key, level + 1, max_level)
            branch_name = ' '.join(sorted(candidate_cluster_key))
            branches[branch_name] = branch_result
    result['branches'] = branches
    return result


def build_keyword_tree(keywords, base_word=None):
    processed = preprocess_keywords(keywords)
    if base_word:
        base_stem = ps.stem(base_word.lower())
    else:
        all_stems = Counter()
        for kw in processed:
            all_stems.update(kw['stems'])
        base_stem = all_stems.most_common(1)[0][0]
    base_data = [kw for kw in processed if base_stem in kw['stems']]
    return cluster_keywords(base_data, {base_stem}, level=1, max_level=3)


def display_tree(tree, level=0):
    indent = " " * level
    cluster_key_display = ', '.join(sorted(tree['cluster_key']))
    st.markdown(f"{indent}**Cluster:** `{cluster_key_display}`")
    if tree.get('exact'):
        for kw in tree['exact']:
            st.markdown(f"{indent}- {kw}")
    if tree.get('extended'):
        for kw in tree['extended']:
            st.markdown(f"{indent}- {kw} _(4+ words)_")
    if 'branches' in tree:
        for _, branch in tree['branches'].items():
            display_tree(branch, level + 1)


def tree_to_rows(tree, parent_path=None):
    """
    Converts the tree structure into flat rows for CSV export.
    """
    if parent_path is None:
        parent_path = []
    rows = []
    current_key = ' '.join(sorted(tree['cluster_key']))
    current_path = parent_path + [current_key]
    path_str = ' > '.join(current_path)

    row = {
        'Cluster Path': path_str,
        'Exact Matches': ', '.join(tree.get('exact', [])),
        'Extended Keywords': ', '.join(tree.get('extended', []))
    }
    rows.append(row)

    if 'branches' in tree:
        for _, branch in tree['branches'].items():
            rows.extend(tree_to_rows(branch, current_path))

    return rows


# ------------------ STREAMLIT APP ------------------

st.title("Keyword Clustering App")

uploaded_file = st.file_uploader("Upload your CSV file with keywords", type=["csv"])
base_word = st.text_input("Optional: Enter a base keyword to cluster (leave blank to auto-select most common)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    column = st.selectbox("Select the keyword column", df.columns)
    keyword_list = df[column].dropna().tolist()

    if st.button("Generate Clusters"):
        with st.spinner("Processing..."):
            tree = build_keyword_tree(keyword_list, base_word=base_word.strip() or None)

            # Display in app
            st.subheader("Clustered Keyword Tree")
            display_tree(tree)

            # Prepare CSV output
            rows = tree_to_rows(tree)
            result_df = pd.DataFrame(rows)

            # Download link for CSV
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Clusters as CSV",
                data=csv,
                file_name='clustered_keywords.csv',
                mime='text/csv'
            )
