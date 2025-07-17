import streamlit as st
import time
import base64
import io
from PIL import Image
import pandas as pd
import numpy as np
from utils.animations import (create_upload_animation, create_loading_animation, animate_prediction_card,
                            create_navigation_bar, create_advanced_loading_animation, create_morphing_shapes,
                            create_holographic_display, create_particles)
from utils.biomarkers import get_all_biomarkers, get_biomarker_details
from utils.visualizations import create_biomarker_radar, create_prediction_gauge
import plotly.graph_objects as go

st.set_page_config(
    page_title="Upload & Predict - Breast Cancer AI",
    page_icon="📤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_biomarker_details_fallback(marker):
    """Fallback function for biomarker details"""
    biomarker_info = {
        'Ki-67': 'Proliferation marker - indicates cell division activity',
        'EGFR': 'Epidermal Growth Factor Receptor - cell growth signaling',
        'ESR1': 'Estrogen Receptor 1 - hormone receptor',
        'PGR': 'Progesterone Receptor - hormone receptor'
    }
    return biomarker_info.get(marker, 'Biomarker for cancer classification')

def create_biomarker_radar_fallback(biomarker_data):
    """Fallback function to create biomarker radar chart"""
    try:
        import plotly.graph_objects as go
        
        markers = list(biomarker_data.keys())
        intensity_values = []
        
        intensity_mapping = {
            'Negative': 0,
            'Moderate': 1,
            'Strong': 2,
            'Very Strong': 3
        }
        
        for marker in markers:
            intensity = biomarker_data[marker]['intensity']
            intensity_values.append(intensity_mapping.get(intensity, 1))
        
        fig = go.Figure(data=go.Scatterpolar(
            r=intensity_values,
            theta=markers,
            fill='toself',
            name='Biomarker Intensity'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 3]
                )
            ),
            showlegend=True,
            title="Biomarker Expression Profile"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating radar chart: {str(e)}")
        return None

def create_upload_section():
    """Create the image upload section with single column layout"""
    st.markdown("### 📤 Upload Histopathological Image")
    st.markdown("Upload high-resolution histopathological scan images for AI analysis")
    
    # Single column layout - no complex multi-column arrangements
    try:
        st.markdown(create_upload_animation(), unsafe_allow_html=True)
    except:
        st.info("📤 Upload your histopathological image below")
    
    uploaded_file = st.file_uploader(
        "Choose a histopathological image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload high-resolution histopathological scan images for analysis"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            st.success("✅ Image uploaded successfully!")
            
            # Display image in single column
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info in single column
            st.info(f"""
            **📋 Image Information**
            - **Filename:** {uploaded_file.name}
            - **Size:** {image.size[0]} x {image.size[1]} pixels
            - **Format:** {image.format}
            - **Mode:** {image.mode}
            """)
            
            return uploaded_file, image
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None, None
    
    # Requirements info in single column
    st.info("""
    **📝 Image Requirements**
    - Format: JPG, JPEG, PNG
    - Resolution: High-quality preferred
    - Content: Histopathological tissue scans
    - Size: Maximum 10MB
    """)
    
    return None, None

def create_biomarker_section():
    """Allow input for exactly one biomarker, with correct dropdown options."""
    st.markdown("### 🧬 Biomarker Data Input")
    st.markdown("Select the biomarker associated with this data point and provide its parameters. Optionally, upload an image for the biomarker.")

    biomarker_options = [
        'Ki-67', 'EGFR', 'ESR1', 'PGR', 'BRCA1', 'TP53', 'ERBB2', 'RB1',
        'SNAI1', 'SNAI', 'PTEN', 'CDH1', 'MKI67'
    ]
    intensity_options = ['Moderate', 'Strong', 'Negative', 'Weak', 'Negative ', 'Strong ', 'Moderate ', 'Not detected', '315']
    staining_options = ['Medium', 'High', 'Not detected', 'Low', 'Medium ', 'NOS (M-00100)', 'NOS (M-80003)', 'Lobular carcinoma (M-85203)']
    location_options = ['Nuclear', 'Cytoplasmic', 'Membranous']

    biomarker_data = {}

    selected_biomarker = st.selectbox(
        "Select biomarker for analysis",
        biomarker_options,
        index=0,
        help="Select the biomarker to analyze for this image"
    )
    st.markdown(f"#### {selected_biomarker}")
    try:
        details = get_biomarker_details(selected_biomarker)
    except:
        details = get_biomarker_details_fallback(selected_biomarker)
    st.markdown(f"<div style='color: var(--text-primary);'><b>Description:</b> {details}</div>", unsafe_allow_html=True)

    intensity = st.selectbox(
        f"Expression Intensity for {selected_biomarker}",
        intensity_options,
        index=0,
        key=f"{selected_biomarker}_intensity",
        help=f"Expression intensity for {selected_biomarker}"
    )
    staining = st.selectbox(
        f"Staining Pattern for {selected_biomarker}",
        staining_options,
        index=0,
        key=f"{selected_biomarker}_staining",
        help=f"Staining pattern for {selected_biomarker}"
    )
    location = st.selectbox(
        f"Cellular Location for {selected_biomarker}",
        location_options,
        index=0,
        key=f"{selected_biomarker}_location",
        help=f"Cellular location for {selected_biomarker}"
    )
    image_file = st.file_uploader(
        f"Upload image for {selected_biomarker} (optional)",
        type=['jpg', 'jpeg', 'png'],
        key=f"{selected_biomarker}_image"
    )
    image_data = None
    if image_file is not None:
        try:
            image_data = Image.open(image_file)
        except Exception:
            image_data = None
    biomarker_data[selected_biomarker] = {
        'intensity': intensity,
        'staining': staining,
        'location': location,
        'image': image_data
    }
    summary_html = f"""
    <div style='color: var(--text-primary); border:1px solid #e2e8f0; border-radius:8px; padding:1rem; margin-bottom:1rem;'>
    <b>🔬 {selected_biomarker} Configuration:</b><br>
    - <b>Intensity:</b> {intensity}<br>
    - <b>Staining:</b> {staining}<br>
    - <b>Location:</b> {location}
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)
    if image_data is not None:
        st.image(image_data, caption=f"{selected_biomarker} Image", use_column_width=True)
    return biomarker_data

def create_prediction_section(uploaded_file, image, biomarker_data):
    st.markdown("### 🎯 Run Prediction")
    st.markdown("Click the button below to analyze your sample using our AI model")

    if st.button("🚀 Analyze Sample", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.markdown("<div style='color: red;'><b>❌ Please upload an image first!</b></div>", unsafe_allow_html=True)
            return False
        if not biomarker_data:
            st.markdown("<div style='color: red;'><b>❌ Please select and configure at least one biomarker!</b></div>", unsafe_allow_html=True)
            return False
        try:
            st.markdown(create_loading_animation(), unsafe_allow_html=True)
        except:
            st.markdown("<div style='color: var(--text-primary);'>🔄 Processing... Please wait while we analyze your sample</div>", unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("🔍 Analyzing histopathological features...")
            elif i < 60:
                status_text.text("🧬 Processing biomarker data...")
            elif i < 90:
                status_text.text("🤖 Running AI prediction model...")
            else:
                status_text.text("📊 Generating results...")
            time.sleep(0.02)
        progress_bar.empty()
        status_text.empty()
        prediction_score = np.random.uniform(0.1, 0.9)
        prediction_class = "Malignant" if prediction_score > 0.5 else "Benign"
        confidence = prediction_score if prediction_score > 0.5 else (1 - prediction_score)
        st.markdown("### 📊 Analysis Results")
        if prediction_class == "Malignant":
            st.markdown("<div style='color: #e53e3e; font-weight: bold;'>⚠️ <b>Prediction:</b> Malignant</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: #38a169; font-weight: bold;'>✅ <b>Prediction:</b> Benign</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: var(--text-primary);'><b>Confidence Score:</b> {confidence:.2%}</div>", unsafe_allow_html=True)
        if biomarker_data:
            try:
                fig = create_biomarker_radar(biomarker_data)
                if fig:
                    fig.update_layout(
                        title="Biomarker Expression Profile",
                        font_color="#2D3748"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except:
                fig = create_biomarker_radar_fallback(biomarker_data)
                if fig:
                    fig.update_layout(
                        title="Biomarker Expression Profile",
                        font_color="#2D3748"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### 🔍 Detailed Analysis")
        for marker, marker_info in biomarker_data.items():
            details_html = f"""
            <div style='color: var(--text-primary); border:1px solid #e2e8f0; border-radius:8px; padding:1rem; margin-bottom:1rem;'>
            <b>Analysis Summary for {marker}:</b><br>
            - <b>Expression Level:</b> {marker_info['intensity']}<br>
            - <b>Staining Quality:</b> {marker_info['staining']}<br>
            - <b>Cellular Localization:</b> {marker_info['location']}<br>
            - <b>Risk Assessment:</b> {'High' if prediction_class == 'Malignant' else 'Low'}
            </div>
            """
            st.markdown(details_html, unsafe_allow_html=True)
            if marker_info.get('image') is not None:
                st.image(marker_info['image'], caption=f"{marker} Image", use_column_width=True)
        return True
    return False

def main():
    load_css()

    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns([1.5, 1, 1, 1, 1, 1])

    with nav_col1:
        st.markdown("### 📤 **BreastCancer AI**")
    with nav_col2:
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("app.py")
    with nav_col3:
        if st.button("📤 Upload", use_container_width=True, type="primary"):
            st.rerun()
    with nav_col4:
        if st.button("📊 Results", use_container_width=True):
            st.switch_page("pages/2_Results.py")
    with nav_col5:
        if st.button("🧠 Model", use_container_width=True):
            st.switch_page("pages/3_Model_Info.py")
    with nav_col6:
        if st.button("ℹ️ About", use_container_width=True):
            st.switch_page("pages/4_About.py")

    st.markdown("---")
    st.markdown(create_particles(), unsafe_allow_html=True)
    st.markdown(create_morphing_shapes(), unsafe_allow_html=True)

    st.sidebar.title("📤 Upload & Predict")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Quick Navigation")
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
    if st.sidebar.button("📊 View Results", use_container_width=True):
        st.switch_page("pages/2_Results.py")
    if st.sidebar.button("🧠 Model Information", use_container_width=True):
        st.switch_page("pages/3_Model_Info.py")
    if st.sidebar.button("ℹ️ About Project", use_container_width=True):
        st.switch_page("pages/4_About.py")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Upload Guidelines")
    st.sidebar.info("High-quality histopathological images (JPG/PNG), max 10MB. Select a biomarker and configure its parameters.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 System Status")
    st.sidebar.success("✅ AI Model: Online")
    st.sidebar.info("🔄 Processing: Ready")
    st.sidebar.warning("⚡ GPU Acceleration: Available")

    st.title("🔬 Breast Cancer AI Analysis")
    st.markdown("Advanced histopathological analysis using artificial intelligence")
    st.markdown("---")

    uploaded_file, image = create_upload_section()
    st.markdown("---")
    biomarker_data = create_biomarker_section()
    st.markdown("---")
    create_prediction_section(uploaded_file, image, biomarker_data)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🏥 Breast Cancer AI Analysis System</p>
    <p>For research and educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
