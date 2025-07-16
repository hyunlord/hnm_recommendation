# Claude Development Guidelines

This document contains project-specific guidelines and context for Claude to assist with development.

## Project Overview

This is an H&M personalized fashion recommendation system implementing various recommendation algorithms using PyTorch Lightning.

## Key Components

### Data
- **articles.csv**: Product metadata (105k products)
- **customers.csv**: Customer information (1.37M customers)
- **transactions_train.csv**: Transaction history (31M transactions)
- **images/**: Product images organized by article ID prefix

### Models to Implement
1. Collaborative Filtering (User-based, Item-based, Matrix Factorization)
2. Content-based Filtering (using product metadata)
3. Neural Collaborative Filtering (NCF)
4. Wide & Deep
5. LightGCN
6. Sequential models (SASRec, GRU4Rec)

### Evaluation Metrics
- Primary: MAP@12 (Mean Average Precision)
- Secondary: Recall@K, Precision@K, NDCG@K

## Development Guidelines

### Code Style
- Use type hints for all functions
- Follow PEP 8 conventions
- Add docstrings to all classes and functions
- Keep functions focused and modular

### PyTorch Lightning Conventions
- Use LightningModule for all models
- Use LightningDataModule for data handling
- Implement proper training_step, validation_step, test_step
- Use torchmetrics for evaluation

### Data Processing
- Handle large datasets efficiently (use chunking/sampling during development)
- Implement proper train/val/test splits based on time
- Consider memory constraints (31M transactions)

### Testing
- Write unit tests for critical functions
- Test data loading and preprocessing
- Validate model outputs

### Performance Optimization
- Use GPU when available
- Implement efficient data loading (multiple workers)
- Consider approximate methods for large-scale operations
- Cache preprocessed data when possible

## Common Commands

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Train a model
python scripts/train.py model=neural_cf

# Start API server
python scripts/serve.py
```

## Important Considerations

1. **Memory Management**: The transaction dataset is ~3.5GB. Use iterative loading or sampling during development.

2. **Time-based Splitting**: Always split data by time for realistic evaluation (last week for test).

3. **Cold Start Problem**: Consider how to handle new users/items.

4. **Scalability**: Design with production deployment in mind.

## References

- [H&M Kaggle Competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
- [PyTorch Lightning Docs](https://pytorch-lightning.readthedocs.io/)
- [RecBole Library](https://recbole.io/) - For reference implementations