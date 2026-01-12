import os
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- CẤU HÌNH HỆ THỐNG ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import streamlit as st
import tempfile
import time

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="ANDEPTRAI OCR",
    page_icon="📄",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Tải mô hình một lần và cache lại"""
    with st.spinner("🚀 Đang tải mô hình vào GPU..."):
        artifact_dict = create_model_dict(device="cuda", dtype=torch.float16)
        converter = PdfConverter(artifact_dict=artifact_dict)
    return converter

def process_pdf(converter, pdf_path):
    """Xử lý OCR cho file PDF"""
    rendered = converter(pdf_path)
    text, _, _ = text_from_rendered(rendered)
    return text

def main():
    st.title("📄 ANDEPTRAI OCR")
    st.markdown("### Chuyển đổi PDF sang văn bản sử dụng AI")
    
    # Tải mô hình
    converter = load_model()
    st.success("✅ Hệ thống sẵn sàng!")
    
    st.markdown("---")
    
    # Upload file
    uploaded_file = st.file_uploader(
        "📁 Tải lên file PDF của bạn",
        type=["pdf"],
        help="Chọn file PDF để chuyển đổi sang văn bản"
    )
    
    # Initialize session state for OCR result
    if 'ocr_result' not in st.session_state:
        st.session_state.ocr_result = None
    if 'ocr_time' not in st.session_state:
        st.session_state.ocr_time = None
    
    if uploaded_file is not None:
        # Lưu file vào session state để hiển thị
        pdf_bytes = uploaded_file.getvalue()
        
        # Hiển thị thông tin file
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📋 **Tên file:** {uploaded_file.name}")
        with col2:
            file_size = len(pdf_bytes) / (1024 * 1024)
            st.info(f"📦 **Kích thước:** {file_size:.2f} MB")
        
        # Nút xử lý
        if st.button("🔍 Bắt đầu OCR", type="primary", use_container_width=True):
            start_time = time.time()
            
            # Tạo file tạm
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_bytes)
                temp_path = tmp_file.name
            
            try:
                with st.spinner("⏳ Đang xử lý OCR... Vui lòng đợi..."):
                    # Thực hiện OCR
                    text = process_pdf(converter, temp_path)
                
                elapsed_time = time.time() - start_time
                
                # Lưu kết quả vào session state
                st.session_state.ocr_result = text
                st.session_state.ocr_time = elapsed_time
                    
            except Exception as e:
                st.error(f"❌ Lỗi xử lý: {str(e)}")
            finally:
                # Xóa file tạm
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Hiển thị kết quả nếu đã xử lý
        if st.session_state.ocr_result is not None:
            text = st.session_state.ocr_result
            elapsed_time = st.session_state.ocr_time
            
            st.success(f"✅ Hoàn thành trong {elapsed_time:.2f} giây!")
            
            st.markdown("---")
            st.markdown("### 📝 Kết quả OCR - So sánh PDF gốc và Markdown")
            
            # Hiển thị song song: PDF gốc và Markdown Preview
            left_col, right_col = st.columns(2)
            
            with left_col:
                st.markdown("#### 📄 File PDF Gốc")
                # Hiển thị PDF trong iframe
                import base64
                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'''
                    <iframe 
                        src="data:application/pdf;base64,{base64_pdf}" 
                        width="100%" 
                        height="600px" 
                        type="application/pdf"
                        style="border: 1px solid #ddd; border-radius: 5px;">
                    </iframe>
                '''
                st.markdown(pdf_display, unsafe_allow_html=True)
            
            with right_col:
                st.markdown("#### 📋 Rendered View")
                # Container với scroll cho rendered markdown
                with st.container(height=600):
                    st.markdown(text)
            
            st.markdown("---")
            
            # Tabs cho raw markdown và download
            st.markdown("### 📝 Raw Markdown Code")
            tab1, tab2 = st.tabs(["📄 Raw Markdown", "📋 Rendered View"])
            
            with tab1:
                st.code(text, language="markdown")
            
            with tab2:
                st.markdown(text)
            
            # Nút tải xuống
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="⬇️ Tải xuống Markdown (.md)",
                    data=text,
                    file_name=f"{uploaded_file.name.replace('.pdf', '')}_ocr.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col_dl2:
                st.download_button(
                    label="⬇️ Tải xuống Text (.txt)",
                    data=text,
                    file_name=f"{uploaded_file.name.replace('.pdf', '')}_ocr.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Thống kê
            st.markdown("---")
            st.markdown("### 📊 Thống kê:")
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("⏱️ Thời gian xử lý", f"{elapsed_time:.2f}s")
            with stat_col2:
                st.metric("📝 Số ký tự", f"{len(text):,}")
            with stat_col3:
                word_count = len(text.split())
                st.metric("📖 Số từ", f"{word_count:,}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🔧 Powered by Marker OCR | GPU Accelerated"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
