FROM python:2.7

# Cập nhật pip
RUN pip install --upgrade pip setuptools wheel

# Cài numpy trước
RUN pip install numpy==1.16.6  # Phiên bản cuối tương thích Python2.7

# Cài các package còn lại
RUN pip install tensorflow==1.12 \
                networkx==2.0 \
                scipy==1.1.0 \
                mahotas==1.4.3 \
                matplotlib==2.2.4 \
                easydict==1.7 \
                scikit-image==0.14.2 \
                scikit-fmm==0.0.9 \
                scikit-learn==0.19.0
