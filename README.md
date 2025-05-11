# ✨ OED Classification Project ✨

This repository contains the codes used in the thesis:
"**A Hybrid Feature Extraction and Explainable AI Framework for Enhanced Diagnosis of Oral Epithelial Dysplasia**"
*DOI: coming soon*

---

## ⬇️ Installation

To run these codes in Jupyter Notebook, you need to install the following Python libraries. The command is also advised to be used on Google Colab, even though some of the libraries have already been installed by default.

You can easily install them using the pip command below:

```bash
pip install numpy pandas scikit-learn matplotlib scipy opencv-python Pillow scikit-image mahotas
```
---

## 📚 Library Breakdown

Here's a quick overview of the essential libraries used in this project:

* **numpy**:
    * 🌟 Purpose: Essential for high-performance numerical computations, especially array and matrix operations.
    * 🎯 Role in Project: Handling numerical data, particularly image pixel values.
    * 💡 Example: Performing element-wise operations on image arrays.

* **pandas**:
    * 🌟 Purpose: Provides powerful, easy-to-use data structures (like DataFrames) and data analysis tools.
    * 🎯 Role in Project: Organizing, manipulating, and loading/saving dataset features.
    * 💡 Example: Loading image feature data from a CSV file into a DataFrame for analysis.

* **scikit-learn** (sklearn):
    * 🌟 Purpose: A comprehensive library for machine learning tasks.
    * 🎯 Role in Project: Implementing classification algorithms, feature scaling, model selection, and evaluation.
    * 💡 Example: Training a `RandomForestClassifier`, scaling features with `StandardScaler`, and evaluating model performance using metrics like accuracy and precision.

* **matplotlib**:
    * 🌟 Purpose: A fundamental 2D plotting library for creating visualizations.
    * 🎯 Role in Project: Generating plots for data exploration, model evaluation, and visualization of results (e.g., feature importances).
    * 💡 Example: Visualizing the importance of different features extracted from images.

* **scipy**:
    * 🌟 Purpose: A library for scientific and technical computing, building on NumPy.
    * 🎯 Role in Project: Utilized for specific scientific algorithms, potentially in optimization or spatial analysis if required by specific methods.
    * 💡 Example: Potentially used for tasks like clustering evaluation (e.g., `linear_sum_assignment` for matching cluster labels).

* **opencv-python** (cv2):
    * 🌟 Purpose: A leading library for computer vision tasks.
    * 🎯 Role in Project: Image loading, basic image manipulation (like thresholding), and potentially contour detection or other image preprocessing steps.
    * 💡 Example: Reading image files from disk and applying image filters.

* **Pillow** (PIL):
    * 🌟 Purpose: The friendly fork of the Python Imaging Library, supporting various image file formats.
    * 🎯 Role in Project: Handling image file operations, such as opening and converting image formats or modes (e.g., to grayscale).
    * 💡 Example: Opening an image file for further processing.

* **scikit-image** (skimage):
    * 🌟 Purpose: A collection of algorithms for image processing, often used for analysis and feature extraction.
    * 🎯 Role in Project: Extracting advanced image features, such as texture descriptors (like GLCM).
    * 💡 Example: Computing the Gray-Level Co-occurrence Matrix for texture analysis.

* **mahotas**:
    * 🌟 Purpose: Another computer vision and image processing library with a focus on speed.
    * 🎯 Role in Project: Extracting specific image features, particularly texture features like Haralick.
    * 💡 Example: Calculating Haralick texture features from image regions.

---

*Find more detailed information on the methodology and results in the full thesis.*