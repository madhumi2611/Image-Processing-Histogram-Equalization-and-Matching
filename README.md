# 🖼️ Image Histogram Processing Studio

A comprehensive Streamlit web application for image histogram processing built from scratch, featuring:
1. **Histogram Equalization**
2. **Histogram Matching (Specification)**

---

## 🚀 Features

- **Quantization Gray Levels ($L$):**
  - Dropdown menu restricted to powers of 2: $2, 4, 8, 16, 32, 64, 128, 256$.
  - Preserves mathematical rigor across any discrete level resolution.

- **Dual Operating Modes:**
  - **📷 Upload Image Mode:**
    - Real digital photograph processing.
    - Full color preservation using **YCrCb** color space (separates luminance $Y$ from chrominance $Cr, Cb$).
    - Isolates $Y$ channel, quantizes/downsamples to selected $L$ levels, executes custom math engine, rescales back to 8-bit $[0, 255]$, and merges back with original untouched $Cr$ and $Cb$ channels.
    - Dynamically takes total pixel count $n = H \times W$ from uploaded images.
    - Supports two independent images for Histogram Matching with differing dimensions and pixel counts.
    - Download button for processed images.
  - **🔢 Manual Numerical Data Entry:**
    - Explicitly specify total number of pixels ($n$).
    - Interactive editable data table for gray levels $r_0 \dots r_{L-1}$.
    - **🎲 Random generation:** Generates random counts summing strictly to $n$.
    - **📖 Textbook Presets:** Standard Gonzalez & Woods $L=8, n=4096$ preset values.
    - Step-by-step mathematical calculation tables:
      - Probability Density Function $P(r_k) = n_k / n$
      - Cumulative Distribution Function $S_k = \sum_{j=0}^k P(r_j)$
      - Scaling $(L-1) \times S_k$
      - Rounding $s_k = \text{round}((L-1) \times S_k)$
      - Level reassignment and combined counts.
    - For matching: Dual entry (Source and Target/Reference), with inverse routing table $s_k \to v_q \to q$ and final redistributed histogram.

- **Unified Dashboard Visualizations (`plot_dashboard_histograms`):**
  - Interactive Plotly bar chart with hover tooltips showing exact pixel count and gray-level value index.
  - Percentage/PDF mode toggle.
  - Fallback option for native Streamlit bar chart.

---

## 📦 Requirements

```bash
pip install streamlit plotly pandas numpy pillow opencv-python-headless
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```
