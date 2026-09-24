# Data and licensing

Every dataset below is included under `data/`, so there is nothing to download. This page credits the original authors and states what the repository's license covers.

## Sources

| Week(s) | File(s) in `data/` | Dataset | Author | License |
|---------|--------------------|---------|--------|---------|
| W1–W3 | `news_data_dedup.csv`, plus `embeddings.joblib` computed from it | [News Headlines 2024](https://www.kaggle.com/datasets/dylanjcastillo/news-headlines-2024) (Kaggle) | Dylan Castillo | MIT |
| W4–W5 | `clothes.csv`, plus `clothes_json.joblib` derived from it | [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) (Kaggle) | Param Aggarwal | MIT |
| W4–W5 | `faq.yaml`, plus `faq.joblib` built from it | FAQ for the fictional store "Fashion Forward Hub", written for this project (replaces the course's FAQ) | This project | Same as this repository |

Notes:

- The W2–W3 notebooks mention the [BBC News](https://www.kaggle.com/datasets/gpreda/bbc-news) dataset (Gabriel Preda, CC0). It is not included, because the notebooks run on `news_data_dedup.csv`.
- The course notebooks don't name the source of the clothing data. Its columns and row count match the Fashion Product Images dataset listed above.
- The news articles and product listings themselves belong to their original publishers and retailer. The datasets are used here for education only.
- Chroma databases (`data/chroma_db/`, `data/chroma_db_products/`) are generated locally and not tracked.

## Copyright and fairness

This is an unofficial, independent adaptation. It is not affiliated with or endorsed by DeepLearning.AI. The course's lessons, videos and original notebooks are not redistributed here; to take the course itself, enrol through DeepLearning.AI.

What this repository contains:

- **Adapted assignment notebooks and code.** These are derived from the course's assignments. The parts that called cloud services (Together.ai, Weaviate Cloud, Arize Phoenix) were rewritten to use local tools (Ollama, Chroma, local no-op tracing), and the notebooks' narrative text was rewritten. The `.py` mirrors keep the original cloud-based code path (`adapted=False`) next to the local one for comparison.
- **Data files under `data/`**, credited above. They are included only so the exercises run offline.

The MIT license in `LICENSE` covers the code written for this adaptation (the local backends, the test suite and the tooling). It does not re-license any course material or third-party data.

**Takedown:** if you hold rights to anything included here and want it removed, please open an issue and it will be taken down.
