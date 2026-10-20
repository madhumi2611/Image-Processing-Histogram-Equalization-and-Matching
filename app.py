"""
app.py - Image Processing Application
Focusing on:
1. Histogram Equalization
2. Histogram Matching (Specification)
Supports:
- Gray Levels (L) restricted to powers of 2 (2, 4, 8, 16, 32, 64, 128, 256)
- Upload Image Mode with YCrCb color space decomposition, downsampling, and reconstruction
- Manual Numerical Data Entry with customizable pixel count, random generation, and preset examples
- Configurable inverse mapping rules: Nearest Value vs Nearest <= s_k
- Unified interactive visualization via plot_dashboard_histograms
"""

import io
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from core_engine import histogram_equalization, histogram_matching
from image_processor import process_image_equalization, process_image_matching
from visualizations import plot_dashboard_histograms
from samples import (
    generate_sample_low_contrast,
    generate_sample_high_contrast,
    generate_sample_portrait
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Histogram Processing Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Minimal UI Styling (Hides Deploy button, header menu, and watermark)
st.markdown("""
<style>
    /* Completely hide Streamlit Deploy button and header decorations */
    .stDeployButton,
    [data-testid="stAppDeployButton"],
    [data-testid="stToolbar"],
    #MainMenu,
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }
    footer {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Clean, modern typography and structure */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.75rem;
        line-height: 1.5;
    }
    .section-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 1.25rem;
    }
    .badge-subtle {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
        padding: 10px 4px;
        color: #64748B;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom-color: #2563EB !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
TEXTBOOK_L8_COUNTS = [790, 1023, 850, 656, 329, 245, 122, 81]  # Sum = 4096
TEXTBOOK_L8_TARGET = [0, 0, 0, 0, 1024, 2048, 1024, 0]          # Sum = 4096

if "eq_manual_n" not in st.session_state:
    st.session_state["eq_manual_n"] = 4096
if "eq_manual_counts" not in st.session_state:
    st.session_state["eq_manual_counts"] = list(TEXTBOOK_L8_COUNTS)

if "match_src_n" not in st.session_state:
    st.session_state["match_src_n"] = 4096
if "match_src_counts" not in st.session_state:
    st.session_state["match_src_counts"] = list(TEXTBOOK_L8_COUNTS)

if "match_tgt_n" not in st.session_state:
    st.session_state["match_tgt_n"] = 4096
if "match_tgt_counts" not in st.session_state:
    st.session_state["match_tgt_counts"] = list(TEXTBOOK_L8_TARGET)


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Processing Parameters")

    # Gray Levels (L) restricted strictly to powers of 2
    L_POWERS = [2, 4, 8, 16, 32, 64, 128, 256]
    selected_L = st.selectbox(
        "Gray Levels (L)",
        options=L_POWERS,
        index=L_POWERS.index(8),
        help="Quantization resolution. Restricted to powers of 2. L=8 is standard for textbook verification; L=256 preserves full photographic depth."
    )

    st.markdown("---")
    st.markdown("### Technique Navigation")
    technique = st.radio(
        "Select Technique",
        options=["Histogram Equalization", "Histogram Matching"],
        help="Equalization spreads luminance uniformly across the full range. Matching conforms the histogram to a target distribution."
    )

    mapping_rule = "nearest"
    if technique == "Histogram Matching":
        st.markdown("---")
        st.markdown("### Inverse Mapping Logic")
        rule_choice = st.selectbox(
            "Routing Rule (G⁻¹(sₖ))",
            options=[
                "Nearest Value (min |vq - sk|)",
                "Nearest ≤ sk (Floor match)",
                "Smallest ≥ sk (Ceiling match)"
            ],
            index=0,
            help=(
                "Nearest Value: standard Gonzalez & Woods (picks min absolute distance |vq - sk|).\n"
                "Nearest <= sk: academic exam convention (picks closest target level vq that does not exceed sk)."
            )
        )
        if "Nearest Value" in rule_choice:
            mapping_rule = "nearest"
        elif "Nearest ≤" in rule_choice:
            mapping_rule = "nearest_le"
        else:
            mapping_rule = "smallest_ge"

    st.markdown("---")
    st.markdown("### Visualization Settings")
    chart_view = st.selectbox("Chart Type", ["Interactive Plotly Bar Chart", "Native Streamlit Bar Chart"])
    show_pct = st.checkbox("Show as Percentages (%)", value=False, help="Display normalized probability percentages instead of raw pixel counts.")

    st.markdown("---")
    st.markdown("### Mathematical Formulation")
    if technique == "Histogram Equalization":
        st.caption(
            "1. $P(r_k) = n_k / n$\n\n"
            "2. $S_k = \\sum_{j=0}^k P(r_j)$\n\n"
            "3. $s_k = \\text{round}((L - 1) \\cdot S_k)$\n\n"
            "4. Combine counts for identical $s_k$ values."
        )
    else:
        st.caption(
            "1. Equalize Source: $r_k \\to s_k$\n\n"
            "2. Equalize Target: $z_q \\to v_q$\n\n"
            "3. Inverse Mapping: $q = G^{-1}(s_k)$\n\n"
            "4. Reassign source counts to matched target level $q$."
        )


# ---------------------------------------------------------
# Helper Functions for Numerical Entry
# ---------------------------------------------------------
def generate_random_counts(total_n: int, L: int) -> List[int]:
    """Generates random counts across L bins that sum strictly to total_n."""
    if total_n <= 0:
        return [0] * L
    probs = np.random.dirichlet(np.ones(L))
    counts = np.random.multinomial(total_n, probs)
    return counts.tolist()


def adapt_counts_to_L(current_counts: List[int], target_L: int, total_n: int) -> List[int]:
    """Ensures count list length matches L while keeping sum close to total_n."""
    curr_len = len(current_counts)
    if curr_len == target_L:
        return current_counts
    return generate_random_counts(total_n, target_L)


def load_preset_scene(choice: str):
    """Loads a built-in synthetic test scene by name."""
    if "Low Contrast" in choice:
        return generate_sample_low_contrast()
    elif "High Contrast" in choice:
        return generate_sample_high_contrast()
    elif "Portrait" in choice:
        return generate_sample_portrait()
    return None


# ---------------------------------------------------------
# Header & Main Mode Tabs
# ---------------------------------------------------------
st.markdown('<div class="app-title">Image Histogram Processing Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Discrete histogram equalization and specification engine with arbitrary gray level quantization '
    f'(L = {selected_L}), YCrCb color space decomposition, and step-by-step mathematical routing.</div>',
    unsafe_allow_html=True
)

tab_upload, tab_manual = st.tabs(["Upload Image Mode", "Manual Numerical Entry"])


# =========================================================
# TAB 1: UPLOAD IMAGE MODE
# =========================================================
PRESET_SCENES = ["None (Use Uploaded)", "Low Contrast Landscape", "High Contrast Sunset", "Portrait Scene"]

with tab_upload:
    st.markdown(f"#### Digital Photograph Processing (L = {selected_L})")
    
    st.markdown(
        f"**Pipeline:** RGB images are transformed to YCrCb space to separate luminance (Y) from chrominance (Cr, Cb). "
        f"The Y channel is quantized to L = {selected_L} levels, processed through the math engine, scaled back to 8-bit (0–255), "
        "and recombined with untouched Cr and Cb channels to prevent color distortion."
    )
    
    with st.expander("Technical Architecture: Color Space Decoupling & Quantization Pipeline", expanded=False):
        st.markdown("**1. Color Space Decoupling (RGB → YCrCb):**")
        st.markdown(
            "Human vision is substantially more sensitive to variations in luminance (brightness) than chromaticity (color). "
            "Performing histogram equalization directly on individual RGB channels alters the relative ratios between channels, causing severe color casts, "
            "hue saturation shifts, and chrominance artifacts. To preserve natural color fidelity, the image is converted to the **YCrCb** color space "
            "(ITU-R BT.601 standard), isolating the achromatic luminance matrix ($Y$) while preserving the color-difference coordinates ($Cr, Cb$) untouched:"
        )
        st.latex(r"""
        \begin{aligned}
        Y &= 0.299R + 0.587G + 0.114B \\
        Cr &= 0.713(R - Y) + 128 \\
        Cb &= 0.564(B - Y) + 128
        \end{aligned}
        """)

        st.markdown(f"**2. Discrete Quantization & Downsampling ($Y \\to Y_L$):**")
        st.markdown(
            f"Standard digital images possess 256 continuous intensity levels $[0, 255]$. To adapt to the selected resolution ($L = {selected_L}$), "
            "the 8-bit luminance matrix is downsampled into discrete quantization bins:"
        )
        st.latex(rf"Y_{{\text{{quantized}}}} = \min\left(\left\lfloor \frac{{Y \cdot {selected_L}}}{{256}} \right\rfloor, {selected_L - 1}\right)")

        st.markdown("**3. Core Mathematical Transformation:**")
        st.markdown(
            "The discrete luminance frequencies are evaluated by the math engine to compute the Probability Density Function ($P(r_k) = n_k / n$), "
            "Cumulative Distribution Function ($S_k = \\sum P(r_j)$), and intensity mapping array ($s_k$ for equalization or $q$ for matching)."
        )

        st.markdown("**4. Dynamic Rescaling & Color Reconstruction (YCrCb → RGB):**")
        st.markdown(
            "The reassigned discrete levels are linearly mapped back to the native 8-bit dynamic range $[0, 255]$:"
        )
        st.latex(rf"Y_{{\text{{upscaled}}}} = \min\left(\text{{round}}\left( \frac{{Y_{{\text{{mapped}}}} \cdot 255}}{{{selected_L - 1}}} \right), 255\right)")
        st.markdown(
            "This enhanced luminance is recombined with the pristine, untouched $Cr$ and $Cb$ chrominance planes and converted back to RGB, "
            "eliminating hue distortion and maintaining color authenticity."
        )

    if technique == "Histogram Equalization":
        st.markdown("##### Histogram Equalization on Photographic Image")
        
        col_src_choice, col_src_sample = st.columns([2, 1])
        with col_src_choice:
            uploaded_file = st.file_uploader(
                "Upload Source Image (PNG, JPG, JPEG, WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
                key="uploader_eq"
            )
        with col_src_sample:
            use_sample = st.selectbox(
                "Or load a preset scene",
                PRESET_SCENES,
                key="sample_eq"
            )

        # Load image
        img_rgb = None
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            img_rgb = np.array(pil_img)
        elif use_sample != "None (Use Uploaded)":
            img_rgb = load_preset_scene(use_sample)

        if img_rgb is not None:
            h, w = img_rgb.shape[:2]
            total_px = h * w
            
            with st.spinner("Processing image via YCrCb color space..."):
                t0 = time.time()
                results = process_image_equalization(img_rgb, selected_L)
                elapsed = (time.time() - t0) * 1000

            # Image Display Row
            st.markdown("##### Visual Comparison")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Original Image** (`{w} × {h}` = `{total_px:,}` pixels)")
                st.image(img_rgb, use_container_width=True)
            with c2:
                st.markdown(f"**Equalized Image (L = {selected_L})** (Processed in `{elapsed:.1f} ms`)")
                st.image(results["output_image"], use_container_width=True)
                
                # Download Button
                out_pil = Image.fromarray(results["output_image"])
                buf = io.BytesIO()
                out_pil.save(buf, format="PNG")
                st.download_button(
                    "Download Equalized Image (PNG)",
                    data=buf.getvalue(),
                    file_name=f"equalized_L{selected_L}.png",
                    mime="image/png"
                )

            # Histograms Row
            st.markdown("##### Luminance Histogram Distribution")
            hist_data = {
                "Original": results["n_k"],
                "Equalized": results["equalized_counts"]
            }

            if chart_view == "Interactive Plotly Bar Chart":
                fig = plot_dashboard_histograms(
                    hist_data,
                    title="Luminance Histogram (Before vs After Equalization)",
                    L=selected_L,
                    show_percentages=show_pct
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                df_bar = plot_dashboard_histograms(
                    hist_data,
                    title="Luminance Histogram",
                    L=selected_L,
                    show_percentages=show_pct
                )
                st.bar_chart(df_bar)

            # Step-by-Step Educational Breakdown
            with st.expander(f"Step-by-Step Mathematical Calculations (L = {selected_L})"):
                st.markdown(
                    f"Total pixels counted from image luminance: $n = {total_px:,}$. "
                    f"Max index level: $(L - 1) = {selected_L - 1}$."
                )
                st.dataframe(results["df_steps"], use_container_width=True)
                st.markdown("###### Equalized Histogram Frequencies")
                st.dataframe(results["df_result"], use_container_width=True)

        else:
            st.info("Upload an image file or select a preset scene above to begin.")

    else:
        # Histogram Matching Mode
        st.markdown("##### Histogram Matching (Specification) on Photographic Images")
        st.markdown(
            "Provide both a **Source Image** (to be modified) and a **Target Reference Image** "
            "(whose luminance distribution will be transferred). Image dimensions and pixel counts can be completely different."
        )

        col_src, col_tgt = st.columns(2)
        with col_src:
            st.markdown("**1. Source Image**")
            src_file = st.file_uploader("Upload Source Image", type=["png", "jpg", "jpeg", "webp"], key="upl_src_match")
            src_sample = st.selectbox(
                "Or load Source Preset",
                PRESET_SCENES,
                index=1,  # Default to Low Contrast Landscape
                key="sample_src_match"
            )
            src_img = None
            if src_file:
                src_img = np.array(Image.open(src_file).convert("RGB"))
            elif src_sample != "None (Use Uploaded)":
                src_img = load_preset_scene(src_sample)

        with col_tgt:
            st.markdown("**2. Target Reference Image**")
            tgt_file = st.file_uploader("Upload Target Reference Image", type=["png", "jpg", "jpeg", "webp"], key="upl_tgt_match")
            tgt_sample = st.selectbox(
                "Or load Target Preset",
                PRESET_SCENES,
                index=2,  # Default to High Contrast Sunset
                key="sample_tgt_match"
            )
            tgt_img = None
            if tgt_file:
                tgt_img = np.array(Image.open(tgt_file).convert("RGB"))
            elif tgt_sample != "None (Use Uploaded)":
                tgt_img = load_preset_scene(tgt_sample)

        if src_img is not None and tgt_img is not None:
            sh, sw = src_img.shape[:2]
            th, tw = tgt_img.shape[:2]

            with st.spinner("Executing Histogram Matching pipeline..."):
                t0 = time.time()
                match_results = process_image_matching(src_img, tgt_img, selected_L, mapping_rule=mapping_rule)
                elapsed = (time.time() - t0) * 1000

            st.markdown("##### Visual Progression")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**Source (Original)**\n`{sw} × {sh}` (`{sh*sw:,}` px)")
                st.image(src_img, use_container_width=True)
            with c2:
                st.markdown(f"**Target (Reference)**\n`{tw} × {th}` (`{th*tw:,}` px)")
                st.image(tgt_img, use_container_width=True)
            with c3:
                st.markdown(f"**Matched Result (L = {selected_L})**\nProcessed in `{elapsed:.1f} ms`")
                st.image(match_results["output_image"], use_container_width=True)
                
                # Download Button
                out_pil = Image.fromarray(match_results["output_image"])
                buf = io.BytesIO()
                out_pil.save(buf, format="PNG")
                st.download_button(
                    "Download Matched Image (PNG)",
                    data=buf.getvalue(),
                    file_name=f"matched_L{selected_L}.png",
                    mime="image/png"
                )

            # Histograms
            st.markdown("##### Comparative Histogram Distribution")
            hist_match_data = {
                "Source Original": match_results["source_res"]["n_k"],
                "Target Specified": match_results["target_res"]["n_k"],
                "Matched Result": match_results["matched_counts"]
            }

            if chart_view == "Interactive Plotly Bar Chart":
                fig = plot_dashboard_histograms(
                    hist_match_data,
                    title=f"Histogram Matching Comparison ({mapping_rule})",
                    L=selected_L,
                    show_percentages=show_pct
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                df_bar = plot_dashboard_histograms(
                    hist_match_data,
                    title="Histogram Matching Comparison",
                    L=selected_L,
                    show_percentages=show_pct
                )
                st.bar_chart(df_bar)

            with st.expander("Mathematical Routing & Equalization Tables"):
                t1, t2, t3 = st.tabs(["Routing / Inverse Mapping", "Source Equalization Table", "Target Equalization Table"])
                with t1:
                    st.dataframe(match_results["df_routing"], use_container_width=True)
                with t2:
                    st.dataframe(match_results["source_res"]["df_steps"], use_container_width=True)
                with t3:
                    st.dataframe(match_results["target_res"]["df_steps"], use_container_width=True)

        else:
            st.warning("Please provide both a Source Image and a Target Reference Image above.")


# =========================================================
# TAB 2: MANUAL NUMERICAL ENTRY
# =========================================================
with tab_manual:
    st.markdown(f"#### Manual Numerical Data Entry (L = {selected_L})")
    st.markdown(
        f"**Pipeline:** Direct numerical simulation mode. Specify the total pixel count ($n$) and assign frequency distributions across $L = {selected_L}$ discrete levels ($r_0 \\dots r_{{selected_L - 1}}$). "
        "Values are evaluated directly by the mathematical engine with no color space transformation required."
    )

    if technique == "Histogram Equalization":
        st.markdown("##### Histogram Equalization - Numerical Simulation")

        col_n, col_actions = st.columns([1, 2])
        with col_n:
            eq_n = st.number_input(
                "Total Pixels (n)",
                min_value=1,
                max_value=10_000_000,
                value=int(st.session_state["eq_manual_n"]),
                step=128,
                key="input_eq_n"
            )
            st.session_state["eq_manual_n"] = eq_n

        with col_actions:
            st.markdown("**Preset & Generator Actions:**")
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("Generate Random Counts", key="btn_eq_random"):
                    st.session_state["eq_manual_counts"] = generate_random_counts(eq_n, selected_L)
                    st.rerun()
            with b_col2:
                if selected_L == 8:
                    if st.button("Load Textbook Preset (n=4096)", key="btn_eq_textbook"):
                        st.session_state["eq_manual_n"] = 4096
                        st.session_state["eq_manual_counts"] = list(TEXTBOOK_L8_COUNTS)
                        st.rerun()
                else:
                    st.caption("*(Textbook preset available when L = 8)*")

        # Synchronize count length with L
        st.session_state["eq_manual_counts"] = adapt_counts_to_L(
            st.session_state["eq_manual_counts"], selected_L, eq_n
        )

        # Interactive Data Editor for Gray Level Counts
        st.markdown(f"###### Enter Pixel Frequencies for Levels $r_0$ to $r_{selected_L - 1}$:")
        
        df_input = pd.DataFrame({
            "Gray Level (r_k)": np.arange(selected_L),
            "Pixel Count (n_k)": st.session_state["eq_manual_counts"]
        })

        edited_df = st.data_editor(
            df_input,
            num_rows="fixed",
            disabled=["Gray Level (r_k)"],
            use_container_width=True,
            key="editor_eq_manual"
        )

        current_counts = edited_df["Pixel Count (n_k)"].to_numpy(dtype=np.int64)
        current_sum = int(np.sum(current_counts))
        st.session_state["eq_manual_counts"] = current_counts.tolist()

        # Validation feedback
        diff = current_sum - eq_n
        if diff == 0:
            st.success(f"Sum of pixel counts exactly matches specified total n = {eq_n:,}.")
        else:
            col_warn, col_fix = st.columns([3, 1])
            with col_warn:
                st.warning(f"Current count sum is {current_sum:,} (difference of {diff:+,} from specified n = {eq_n:,}).")
            with col_fix:
                if st.button("Auto-Scale to Match n"):
                    if current_sum > 0:
                        scaled = np.round((current_counts / current_sum) * eq_n).astype(np.int64)
                        scaled[-1] += (eq_n - int(np.sum(scaled)))
                        st.session_state["eq_manual_counts"] = scaled.tolist()
                        st.rerun()

        # Execute Engine
        eq_res = histogram_equalization(current_counts, selected_L)

        # Plot Histograms
        st.markdown("##### Interactive Distribution Visualization")
        eq_chart_dict = {
            "Original": eq_res["n_k"],
            "Equalized": eq_res["equalized_counts"]
        }
        if chart_view == "Interactive Plotly Bar Chart":
            fig = plot_dashboard_histograms(
                eq_chart_dict,
                title="Manual Histogram Equalization Distribution",
                L=selected_L,
                show_percentages=show_pct
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            df_bar = plot_dashboard_histograms(
                eq_chart_dict,
                title="Manual Histogram Equalization",
                L=selected_L,
                show_percentages=show_pct
            )
            st.bar_chart(df_bar)

        # Step-by-Step Numerical Calculations Table
        st.markdown("##### Step-by-Step Mathematical Calculation Table")
        st.dataframe(eq_res["df_steps"], use_container_width=True)

        st.markdown("##### Equalized Histogram Result")
        st.dataframe(eq_res["df_result"], use_container_width=True)

    else:
        # Histogram Matching Numerical Mode
        st.markdown("##### Histogram Matching (Specification) - Numerical Simulation")
        st.markdown(
            "Configure two independent numerical histograms: **Source** and **Target Specified**. "
            "The total number of pixels ($n_1$ and $n_2$) can differ completely."
        )

        col_src_card, col_tgt_card = st.columns(2)

        # Source Section
        with col_src_card:
            st.markdown("**1. Source Histogram Entry**")
            src_n = st.number_input(
                "Source Total Pixels (n_source)",
                min_value=1,
                max_value=10_000_000,
                value=int(st.session_state["match_src_n"]),
                step=128,
                key="input_match_src_n"
            )
            st.session_state["match_src_n"] = src_n

            sb1, sb2 = st.columns(2)
            with sb1:
                if st.button("Generate Random Source", key="btn_src_rand"):
                    st.session_state["match_src_counts"] = generate_random_counts(src_n, selected_L)
                    st.rerun()
            with sb2:
                if selected_L == 8:
                    if st.button("Load Preset Source", key="btn_src_preset"):
                        st.session_state["match_src_n"] = 4096
                        st.session_state["match_src_counts"] = list(TEXTBOOK_L8_COUNTS)
                        st.rerun()

            st.session_state["match_src_counts"] = adapt_counts_to_L(
                st.session_state["match_src_counts"], selected_L, src_n
            )

            df_src_input = pd.DataFrame({
                "Level (k)": np.arange(selected_L),
                "Source Count (n_k)": st.session_state["match_src_counts"]
            })
            edited_src_df = st.data_editor(
                df_src_input,
                num_rows="fixed",
                disabled=["Level (k)"],
                use_container_width=True,
                key="editor_match_src"
            )
            src_counts = edited_src_df["Source Count (n_k)"].to_numpy(dtype=np.int64)
            st.session_state["match_src_counts"] = src_counts.tolist()
            st.caption(f"Source Sum: **{int(np.sum(src_counts)):,}** / Specified Total: **{src_n:,}**")

        # Target Section
        with col_tgt_card:
            st.markdown("**2. Target Reference Histogram Entry**")
            tgt_n = st.number_input(
                "Target Total Pixels (n_target)",
                min_value=1,
                max_value=10_000_000,
                value=int(st.session_state["match_tgt_n"]),
                step=128,
                key="input_match_tgt_n"
            )
            st.session_state["match_tgt_n"] = tgt_n

            tb1, tb2 = st.columns(2)
            with tb1:
                if st.button("Generate Random Target", key="btn_tgt_rand"):
                    st.session_state["match_tgt_counts"] = generate_random_counts(tgt_n, selected_L)
                    st.rerun()
            with tb2:
                if selected_L == 8:
                    if st.button("Load Preset Target", key="btn_tgt_preset"):
                        st.session_state["match_tgt_n"] = 4096
                        st.session_state["match_tgt_counts"] = list(TEXTBOOK_L8_TARGET)
                        st.rerun()

            st.session_state["match_tgt_counts"] = adapt_counts_to_L(
                st.session_state["match_tgt_counts"], selected_L, tgt_n
            )

            df_tgt_input = pd.DataFrame({
                "Level (q)": np.arange(selected_L),
                "Target Count (n_q)": st.session_state["match_tgt_counts"]
            })
            edited_tgt_df = st.data_editor(
                df_tgt_input,
                num_rows="fixed",
                disabled=["Level (q)"],
                use_container_width=True,
                key="editor_match_tgt"
            )
            tgt_counts = edited_tgt_df["Target Count (n_q)"].to_numpy(dtype=np.int64)
            st.session_state["match_tgt_counts"] = tgt_counts.tolist()
            st.caption(f"Target Sum: **{int(np.sum(tgt_counts)):,}** / Specified Total: **{tgt_n:,}**")

        # Compute Matching with selected rule
        match_res = histogram_matching(src_counts, tgt_counts, selected_L, mapping_rule=mapping_rule)

        st.markdown("---")
        st.markdown(f"##### Comparative Histogram Distribution (Active Rule: `{mapping_rule}`)")
        match_chart_dict = {
            "Source Original": src_counts,
            "Target Specified": tgt_counts,
            "Matched Result": match_res["matched_counts"]
        }

        if chart_view == "Interactive Plotly Bar Chart":
            fig = plot_dashboard_histograms(
                match_chart_dict,
                title=f"Histogram Matching Distributions ({mapping_rule})",
                L=selected_L,
                show_percentages=show_pct
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            df_bar = plot_dashboard_histograms(
                match_chart_dict,
                title="Histogram Matching Distributions",
                L=selected_L,
                show_percentages=show_pct
            )
            st.bar_chart(df_bar)

        # Routing and Detailed Breakdown
        st.markdown(f"##### Step 3: Inverse Mapping & Routing Table ($s_k \\to v_q \\to q$)")
        st.markdown(
            f"Active rule: **`{mapping_rule}`**. "
            "For each source equalized value $s_k$, the closest target level $q$ is matched according to the selected criterion."
        )
        st.dataframe(match_res["df_routing"], use_container_width=True)

        with st.expander("View Source and Target Equalization Math Tables"):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("###### Source Equalization Steps")
                st.dataframe(match_res["source_res"]["df_steps"], use_container_width=True)
            with sc2:
                st.markdown("###### Target Equalization Steps")
                st.dataframe(match_res["target_res"]["df_steps"], use_container_width=True)

        st.markdown("##### Final Redistributed Histogram")
        st.dataframe(match_res["df_result"], use_container_width=True)
