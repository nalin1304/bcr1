import streamlit as st
import time, base64, io, pickle
from PIL import Image
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Your existing utils
from utils.animations import (
    create_upload_animation, create_loading_animation, animate_prediction_card,
    create_navigation_bar, create_advanced_loading_animation, create_morphing_shapes,
    create_holographic_display, create_particles
)
from utils.biomarkers import get_all_biomarkers, get_biomarker_details
from utils.visualizations import create_biomarker_radar, create_prediction_gauge

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoModel

# -----------------------------------------------------------------------------
# 1) MODEL DEFINITION & LOADING
# -----------------------------------------------------------------------------

class DinoMLPFusion(nn.Module):
    def __init__(self, num_classes, bio_dim):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("1aurent/vit_small_patch16_256.tcga_brca_dino")
        for p in self.backbone.parameters(): p.requires_grad = False
        vit_out_dim = 384
        self.bio_proj = nn.Linear(bio_dim, 128)
        self.head = nn.Sequential(
            nn.Linear(vit_out_dim + 128, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, img, bio):
        with torch.no_grad():
            x_img = self.backbone(img).pooler_output
        x_bio = self.bio_proj(bio)
        return self.head(torch.cat([x_img, x_bio], dim=1))

@st.cache_resource
def load_model_and_mappings():
    # 1.1 Load label mapping
    with open("lbl_map.pkl","rb") as f: lbl_map = pickle.load(f)
    idx_to_subtype = {v:k for k,v in lbl_map.items()}
    # 1.2 Load biomarker feature columns order
    with open("bm_feat_cols.pkl","rb") as f: bm_feat_cols = pickle.load(f)
    # 1.3 Build model
    model = DinoMLPFusion(num_classes=len(lbl_map), bio_dim=len(bm_feat_cols))
    model.load_state_dict(torch.load("dino_model.pth", map_location="cpu"))
    model.eval()
    return model, bm_feat_cols, idx_to_subtype

model, bm_feat_cols, idx_to_subtype = load_model_and_mappings()

# -----------------------------------------------------------------------------
# 2) PREPROCESSING HELPERS
# -----------------------------------------------------------------------------

normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
transform_dino = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(), normalize
])

def process_biomarker_input(marker, intensity, staining, bm_feat_cols):
    user = {
        f"biomarker_{marker}":1.0,
        f"intensity_{intensity}":1.0,
        f"staining_{staining}":1.0
    }
    vec = [user.get(col,0.0) for col in bm_feat_cols]
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

def predict_dino(model, img, bio_vec):
    img_t = transform_dino(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_t, bio_vec)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
    return idx, probs

# -----------------------------------------------------------------------------
# 3) STREAMLIT CONFIG & UTILS
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer AI", page_icon="🧬", layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_biomarker_details_fallback(m):
    info = {
        'Ki-67':'Proliferation marker',
        'EGFR':'Epidermal Growth Factor Receptor',
        'ESR1':'Estrogen Receptor 1',
        'PGR':'Progesterone Receptor'
    }
    return info.get(m, "Biomarker for classification")

def radar_fallback(data):
    try:
        markers=list(data.keys()); vals=[]
        mp={'Negative':0,'Moderate':1,'Strong':2,'Very Strong':3}
        for m in markers: vals.append(mp.get(data[m]['intensity'],1))
        fig=go.Figure(data=go.Scatterpolar(r=vals,theta=markers,fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,3])),showlegend=False)
        return fig
    except Exception as e:
        st.error(f"Radar error: {e}")
        return None

# -----------------------------------------------------------------------------
# 4) UI COMPONENTS
# -----------------------------------------------------------------------------

def create_upload_section():
    st.markdown("### 📤 Upload Histopathological Image")
    st.markdown("High-res tissue scans (JPG/PNG, ≤10MB)")
    try: st.markdown(create_upload_animation(), unsafe_allow_html=True)
    except: st.info("Upload your image below")
    f=st.file_uploader("", type=['jpg','jpeg','png'])
    if f:
        try:
            img=Image.open(f); st.success("Image loaded")
            st.image(img, use_column_width=True)
            st.info(f"Filename: {f.name} • Size: {img.size[0]}×{img.size[1]} px")
            return f,img
        except: st.error("Invalid image"); return None,None
    st.info("⛔ Please upload an image to proceed")
    return None,None

def create_biomarker_section():
    st.markdown("### 🧬 Biomarker Data Input")
    opts = ['Ki-67','EGFR','ESR1','PGR','BRCA1','TP53','ERBB2','RB1','SNAI1','SNAI','PTEN','CDH1','MKI67']
    i_opts = ['Moderate','Strong','Negative','Weak','Negative ','Strong ','Moderate ','Not detected','315']
    s_opts = ['Medium','High','Not detected','Low','Medium ','NOS (M-00100)','NOS (M-80003)','Lobular carcinoma (M-85203)']
    data={}
    m=st.selectbox("Biomarker", opts)
    try: det=get_biomarker_details(m)
    except: det=get_biomarker_details_fallback(m)
    st.markdown(f"**Description:** {det}")
    intensity=st.selectbox("Intensity", i_opts)
    staining=st.selectbox("Staining", s_opts)
    data[m]={'intensity':intensity,'staining':staining,'location':None,'image':None}
    st.markdown(f"<div style='padding:1rem;border:1px solid #e2e8f0;border-radius:8px;'><b>{m}:</b> {intensity}, {staining}</div>", unsafe_allow_html=True)
    return data

def create_prediction_section(uploaded_file, image, biomarker_data):
    st.markdown("### 🎯 Run Prediction")
    if st.button("🚀 Analyze Sample", type="primary"):
        if not uploaded_file: st.error("Upload an image first!"); return
        if not biomarker_data: st.error("Configure a biomarker!"); return

        # Animated loading
        try: st.markdown(create_loading_animation(), unsafe_allow_html=True)
        except: st.info("Analyzing…")
        p=st.progress(0); t=st.empty()
        for i in range(100):
            p.progress(i+1)
            t.text(["🔍","🧬","🤖","📊"][i//25]+" …")
            time.sleep(0.01)
        p.empty(); t.empty()

        # Real inference
        mk=next(iter(biomarker_data))
        iv=biomarker_data[mk]['intensity']; sv=biomarker_data[mk]['staining']
        vec=process_biomarker_input(mk, iv, sv, bm_feat_cols)
        idx,probs=predict_dino(model, image, vec)
        sub=idx_to_subtype[idx]; conf=probs[idx]

        st.success(f"🧬 Predicted Subtype: **{sub}**")
        st.info(f"Confidence: **{conf:.1%}**")

        dfp=pd.DataFrame({
            'Subtype':[idx_to_subtype[i] for i in range(len(probs))],
            'Prob':[f"{p:.1%}" for p in probs]
        })
        st.table(dfp)

        # Radar
        try: fig=create_biomarker_radar(biomarker_data)
        except: fig=radar_fallback(biomarker_data)
        if fig: st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 5) MAIN APP
# -----------------------------------------------------------------------------

def main():
    load_css()
    # top navigation
    nav = create_navigation_bar()  # assuming it builds your 6-button row
    st.markdown(create_morphing_shapes(), unsafe_allow_html=True)
    st.sidebar.title("🔗 Quick Nav")
    for name,page in [("Home","app.py"),("Upload","app.py"),("Results","pages/2_Results.py"),
                     ("Model","pages/3_Model_Info.py"),("About","pages/4_About.py")]:
        if st.sidebar.button(name): st.switch_page(page)
    st.sidebar.markdown("---")
    st.sidebar.info("Upload histopathology image & biomarker data")
    st.sidebar.success("AI Model: Online")
    st.sidebar.warning("GPU: Available")

    st.title("🔬 Breast Cancer AI Analysis")
    st.markdown("---")

    uploaded_file, image = create_upload_section()
    st.markdown("---")
    biomarker_data      = create_biomarker_section()
    st.markdown("---")
    create_prediction_section(uploaded_file, image, biomarker_data)

    st.markdown("---")
    st.markdown("<center>🏥 For research use only</center>", unsafe_allow_html=True)

if __name__=="__main__":
    main()
