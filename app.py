import streamlit as st
import cv2
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import imutils
import os
from PIL import Image


css = """
<style>
    [data-testid='stSidebarNav'] > ul {
        min-height: 50vh;
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Streamlit app
st.title("Ουσίες, Κουκίδες & οινοπνεύματα..")
st.write(
    "Πανε στο more.com η στο τικετμαστερ, βγαλε screenshot τις κουκιδες ξεχωριστα για καθε κερκίδα και ανεβασε τες εδω. Απο κατω θα σου βγει ποσες θεσις καθε χρώματος ειναι ελευθερες, ενω με γκρί οσες εχουν αγοραστεί "
)

logo_image_path = "ppsks.jpg"
if os.path.exists(logo_image_path):
    st.sidebar.image(logo_image_path, width=150)
else:
    st.sidebar.text("Εδώ θα ηταν φωτο Παπασάκη.")


# Sidebar
st.sidebar.title("551")

st.sidebar.write(
    "Μονε σε αγαπω και ας με πληγωσες γιατι οσοι αγαπανε πονανε και οσοι πονανε αγαπανε. Οσοι πονανε σιγουρα πονανε και οσοι αγαπανε αγαπανε. Εκτος απο εσενα Μ.Ι."
)

# File uploader
uploaded_files = st.file_uploader(
    "Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if st.sidebar.button("Show Overlay"):
    overlay_image_path = "kgs.jpg"  # Replace with your overlay image path
    if os.path.exists(overlay_image_path):
        st.sidebar.image(
            overlay_image_path,
            caption="Με εμενα θα ειχατε 5 ευρω γενική",
            use_column_width=True,
        )
        st.sidebar.write("λευτερία στον Θ.Δ")
    else:
        st.sidebar.text("Εδώ θα ηταν φωτο δικηγόρου over 1.83.")


seats_cnt = 0

if uploaded_files:
    color_region_counts = defaultdict(int)

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image = np.array(image)

        gray_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray_scaled, 30, 150)
        thresh = cv2.threshold(gray_scaled, 225, 225, cv2.THRESH_BINARY_INV)[1]
        contours = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = imutils.grab_contours(contours)

        output = image.copy()
        seats_cnt += len(contours)

        for contour in contours:
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            pixels = masked_image[mask == 255]

            pixels = [tuple(p) for p in pixels]

            color_frequency = Counter(pixels)

            dominant_color = color_frequency.most_common(1)[0][0]

            color_region_counts[dominant_color] += 1

    fig, ax = plt.subplots()
    circle_radius = 0.5

    for idx, (color, count) in enumerate(color_region_counts.items()):
        normalized_color = [int(c) / 255.0 for c in color]

        circle = plt.Circle((1, idx * 2), circle_radius, color=normalized_color)
        ax.add_artist(circle)

        plt.text(2, idx * 2, f"Count: {count}", verticalalignment="center", fontsize=12)

    ax.set_xlim(0, 5)
    ax.set_ylim(-1, len(color_region_counts) * 2)
    ax.set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.axis("off")

    st.pyplot(fig)

    st.write(f"{seats_cnt} συνολικές θέσεις.")
