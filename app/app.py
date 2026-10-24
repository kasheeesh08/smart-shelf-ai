
import json
import streamlit as st
from PIL import Image
from inference import detect_products

DISPLAY_NAMES = {
    0: "Bournvita",
    1: "Mysore Sandal Soap",
    2: "Nivea Body Lotion",
    3: "Nivea Soft Cream",
    4: "Parachute Coconut Oil",
    5: "Patanjali Dant Kanti",
    6: "Society Tea Powder",
    7: "Tresemme Conditioner",
    8: "Tresemme Shampoo",
}

st.set_page_config(
    page_title="Smart Shelf AI",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
h1 {letter-spacing: -1px;}
[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.2);
    padding: 16px; border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("🛒 Smart Shelf AI")
st.caption("AI-powered product detection and shelf inventory counting.")

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Confidence threshold", 0.10, 0.90, 0.45, 0.05,
        help="Higher confidence reduces uncertain detections."
    )
    st.caption("Model: YOLOv8n · 9 product classes")
    st.caption("Results are estimates and may miss or misclassify products.")

uploaded = st.file_uploader(
    "Upload a shelf or product image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Analyzing image..."):
        annotated, detections, counts = detect_products(
            image, confidence=confidence
        )

    st.divider()
    left, right = st.columns([1.65, 1], gap="large")

    with left:
        st.subheader("Detection preview")
        st.image(annotated, use_container_width=True)
        st.download_button(
            "Download annotated image",
            data=__import__("io").BytesIO(
                __import__("PIL").Image.Image.tobytes(annotated)
            ).getvalue(),
            file_name="annotated_shelf.rgb",
            mime="application/octet-stream",
            disabled=True,
            help="Use the detection JSON download for now."
        )

    with right:
        st.subheader("Inventory summary")
        st.metric("Total detected products", len(detections))
        st.metric("Unique product types", len(counts))

        if counts:
            st.markdown("#### Product counts")
            for product, count in sorted(counts.items()):
                st.write(f"**{product}**")
                st.progress(min(count / max(counts.values()), 1.0),
                            text=f"{count} detected")
        else:
            st.info("No products detected. Try lowering the confidence threshold.")

        st.download_button(
            "Download detections (JSON)",
            data=json.dumps({
                "image": uploaded.name,
                "confidence_threshold": confidence,
                "total_detections": len(detections),
                "product_counts": counts,
                "detections": detections
            }, indent=2),
            file_name="shelf_detections.json",
            mime="application/json",
            use_container_width=True
        )

    with st.expander("View individual detections"):
        if detections:
            table = [
                {
                    "Product": d["product"],
                    "Confidence": f'{d["confidence"]:.1%}',
                    "Bounding box (x1, y1, x2, y2)": str(d["bbox_xyxy"])
                }
                for d in detections
            ]
            st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.write("No detections to display.")

else:
    st.info("Upload an image to begin product detection.")
    st.markdown("**Supported classes:** " + ", ".join(DISPLAY_NAMES.values()))
