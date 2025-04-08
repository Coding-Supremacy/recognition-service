import io
import streamlit as st
import boto3
from PIL import Image, ImageDraw, ImageFont

# 🔐 Streamlit secrets에서 AWS 자격증명 불러오기
aws_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["AWS_SECRET_ACCESS_KEY"]
region = "ap-northeast-2"  # 서울 리전

# ▶️ Rekognition 클라이언트 생성
client = boto3.client(
    'rekognition',
    aws_access_key_id=aws_key,
    aws_secret_access_key=aws_secret,
    region_name=region
)

# 사이드바 메뉴 설정
st.sidebar.title("메뉴 선택")
app_mode = st.sidebar.selectbox("기능을 선택하세요", ["얼굴 감정 분석", "얼굴 비교"])

def draw_faces_with_info(image_bytes, face_details):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for idx, face in enumerate(face_details):
        box = face['BoundingBox']
        left = int(box['Left'] * width)
        top = int(box['Top'] * height)
        box_width = int(box['Width'] * width)
        box_height = int(box['Height'] * height)
        
        box_colors = ["red", "blue", "green", "yellow", "purple"]
        box_color = box_colors[idx % len(box_colors)]
        
        draw.rectangle(
            [(left, top), (left + box_width, top + box_height)],
            outline=box_color, width=3
        )
        
        gender = face['Gender']['Value']
        age_range = face['AgeRange']
        emotions = face['Emotions']
        top_emotion = max(emotions, key=lambda x: x['Confidence'])
        
        info_text = (
            f"얼굴 {idx+1}\n"
            f"성별: {gender}\n"
            f"나이: {age_range['Low']}-{age_range['High']}\n"
            f"감정: {top_emotion['Type']}"
        )
        
        text_bg_height = 80
        draw.rectangle(
            [(left, top - text_bg_height), (left + 150, top)],
            fill="black"
        )
        draw.text(
            (left + 5, top - text_bg_height + 5),
            info_text,
            fill=box_color,
            font=font
        )
    
    return image

if app_mode == "얼굴 감정 분석":
    st.title("🧑‍🔬 얼굴 인식기 (Amazon Rekognition 기반 감정 분석)")
    
    uploaded_file = st.file_uploader("얼굴이 보이는 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()

        response = client.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['ALL']
        )

        face_details = response['FaceDetails']

        if len(face_details) == 0:
            st.warning("얼굴이 감지되지 않았습니다.")
            st.image(image_bytes, caption="업로드한 이미지", use_container_width=True)
        else:
            annotated_image = draw_faces_with_info(image_bytes, face_details)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_bytes, caption="원본 이미지", use_container_width=True)
            with col2:
                st.image(annotated_image, caption="얼굴 분석 결과", use_container_width=True)

            st.subheader("👤 상세 얼굴 분석 결과")
            
            for idx, face in enumerate(face_details):
                st.markdown(f"### 🟢 얼굴 {idx+1} 분석 결과")
                
                box_colors = ["red", "blue", "green", "yellow", "purple"]
                box_color = box_colors[idx % len(box_colors)]
                st.markdown(f"<hr style='border: 2px solid {box_color}'>", unsafe_allow_html=True)
                
                age_range = face['AgeRange']
                gender = face['Gender']
                emotions = face['Emotions']
                top_emotion = max(emotions, key=lambda x: x['Confidence'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("성별", f"{gender['Value']} ({gender['Confidence']:.1f}%)")
                with col2:
                    st.metric("추정 나이", f"{age_range['Low']}~{age_range['High']}세")
                with col3:
                    st.metric("주요 감정", f"{top_emotion['Type']} ({top_emotion['Confidence']:.1f}%)")
                
                st.write("**감정 분석 결과:**")
                emotions_sorted = sorted(emotions, key=lambda x: -x['Confidence'])
                for emotion in emotions_sorted:
                    st.progress(int(emotion['Confidence']), 
                              text=f"{emotion['Type']}: {emotion['Confidence']:.1f}%")
                
                with st.expander("추가 얼굴 속성 보기"):
                    attrs = {
                        "Eyeglasses": "안경 착용",
                        "Sunglasses": "선글라스 착용",
                        "Beard": "수염",
                        "Mustache": "콧수염",
                        "EyesOpen": "눈 뜸",
                        "MouthOpen": "입 열림",
                        "Smile": "미소"
                    }
                    
                    for attr, desc in attrs.items():
                        if attr in face:
                            value = face[attr]
                            st.write(f"- {desc}: {'예' if value['Value'] else '아니오'} ({value['Confidence']:.1f}%)")
                
                st.write("")

elif app_mode == "얼굴 비교":
    st.title("🧑‍🤝‍🧑 얼굴 비교기 (Amazon Rekognition)")

    st.subheader("📷 비교할 두 장의 얼굴 사진을 업로드하세요")
    col1, col2 = st.columns(2)
    with col1:
        image1 = st.file_uploader("이미지 1", type=["jpg", "jpeg", "png"], key="img1")
    with col2:
        image2 = st.file_uploader("이미지 2", type=["jpg", "jpeg", "png"], key="img2")

    if image1 and image2:
        image1_bytes = image1.read()
        image2_bytes = image2.read()

        try:
            response = client.compare_faces(
                SourceImage={'Bytes': image1_bytes},
                TargetImage={'Bytes': image2_bytes},
                SimilarityThreshold=50
            )

            st.subheader("🔍 비교 결과")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image1_bytes, caption="이미지 1", use_container_width=True)
            with col2:
                st.image(image2_bytes, caption="이미지 2", use_container_width=True)

            face_matches = response['FaceMatches']
            if len(face_matches) == 0:
                st.warning("두 사람은 동일 인물로 인식되지 않았습니다.")
            else:
                match = face_matches[0]
                similarity = match['Similarity']
                
                st.write(f"**얼굴 유사도:** {similarity:.2f}%")
                st.progress(int(similarity), text=f"유사도 {similarity:.2f}%")
                
                if similarity > 95:
                    st.success("✅ 같은 사람일 가능성이 매우 높습니다 (95% 이상)")
                elif similarity > 80:
                    st.info("🟢 같은 사람일 가능성이 있습니다 (80%~95%)")
                elif similarity > 60:
                    st.warning("🟡 유사하지만 다른 사람일 수 있습니다 (60%~80%)")
                else:
                    st.error("🔴 다른 사람일 가능성이 높습니다 (60% 미만)")

            with st.expander("기술적 세부 정보 보기"):
                st.json(response)

        except Exception as e:
            st.error(f"에러 발생: {str(e)}")
