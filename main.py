import streamlit as st

# Thiết lập cấu hình giao diện
st.set_page_config(page_title="Đăng ký tài khoản", page_icon="🛒", layout="centered")

# CSS cho nền xanh dương nhạt
st.markdown(
    """
    <style>
    body {
        background-color: #e6f0ff;
    }
    .main {
        background-color: #e6f0ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Tiêu đề
st.title("🛍️ Đăng ký tài khoản mua hàng")

# Mô tả nhỏ
st.write("Vui lòng điền thông tin bên dưới để tạo tài khoản của bạn.")

# Form đăng ký
with st.form("registration_form"):
    full_name = st.text_input("Họ và tên")
    email = st.text_input("Email")
    username = st.text_input("Tên đăng nhập")
    password = st.text_input("Mật khẩu", type="password")
    confirm_password = st.text_input("Xác nhận mật khẩu", type="password")
    address = st.text_area("Địa chỉ giao hàng")
    agree = st.checkbox("Tôi đồng ý với các điều khoản & chính sách")

    submitted = st.form_submit_button("Đăng ký")

    if submitted:
        if not agree:
            st.warning("Bạn cần đồng ý với điều khoản và chính sách.")
        elif password != confirm_password:
            st.error("Mật khẩu xác nhận không khớp.")
        elif not all([full_name, email, username, password, confirm_password, address]):
            st.error("Vui lòng điền đầy đủ tất cả các trường.")
        else:
            st.success(f"Tài khoản '{username}' đã được tạo thành công!")
