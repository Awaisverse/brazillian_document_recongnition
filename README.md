# Brazilian Document Classification

Classify Brazilian document types (CNH, CPF, RG, etc.) using a ResNet18 model. **Plan:** 60% training, 40% held-out testing.

## Kaggle Notebook

Use `brazilian_document_classifier.ipynb` on Kaggle:

1. Upload the notebook to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Add your dataset as Input (or use a public Brazilian document dataset)
3. Set `DATA_PATH` in the notebook (e.g. `/kaggle/input/your-dataset-name`)
4. Enable **GPU** in Settings → Accelerator
5. Run all cells

### Dataset Structure

```
your-dataset/
  CNH/        # National Driver's License
  CPF/        # Tax ID document
  RG/         # General Registration
  ...         # 8–9 document types total
```

Each folder = one document type, containing images (`.jpg`, `.png`, etc.).

### Output

- Trained model saved to `/kaggle/working/document_classifier.pt`
- Test accuracy reported on the 40% held-out set

## License

MIT
