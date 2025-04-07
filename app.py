import io
import streamlit as st
import boto3
from PIL import Image, ImageDraw, ImageFont

# Rekognition 클라이언트 생성
client = boto3.client('rekognition', region_name='ap-northeast-2')

# 사이드바 메뉴 설정
st.sidebar.title("메뉴 선택")
app_mode = st.sidebar.selectbox(
    "기능을 선택하세요",
    ["얼굴 감정 분석", "얼굴 비교"]
)

def draw_faces_with_info(image_bytes, face_details):
    """얼굴 박스와 정보를 이미지에 그리는 함수"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # 간단한 폰트 생성 (Streamlit Cloud에서는 기본 폰트 사용)
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
        
        # 박스 색상 (각 얼굴마다 다른 색상 사용)
        box_colors = ["red", "blue", "green", "yellow", "purple"]
        box_color = box_colors[idx % len(box_colors)]
        
        # 얼굴 박스 그리기
        draw.rectangle(
            [(left, top), (left + box_width, top + box_height)],
            outline=box_color, width=3
        )
        
        # 얼굴 정보 요약
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
        
        # 텍스트 배경 (가독성 향상을 위해)
        text_bg_height = 80
        draw.rectangle(
            [(left, top - text_bg_height), (left + 150, top)],
            fill="black"
        )
        
        # 얼굴 정보 텍스트
        draw.text(
            (left + 5, top - text_bg_height + 5),
            info_text,
            fill=box_color,
            font=font
        )
    
    return image

if app_mode == "얼굴 감정 분석":
    st.title("🧑‍🔬 얼굴 인식기 (Amazon Rekognition 기반 감정 분석)")
    
    # 이미지 업로드
    uploaded_file = st.file_uploader("얼굴이 보이는 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()

        # Rekognition 얼굴 감지
        response = client.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['ALL']
        )

        face_details = response['FaceDetails']

        if len(face_details) == 0:
            st.warning("얼굴이 감지되지 않았습니다.")
            st.image(image_bytes, caption="업로드한 이미지", use_container_width=True)
        else:
            # 얼굴 박스와 정보가 포함된 이미지 생성
            annotated_image = draw_faces_with_info(image_bytes, face_details)
            
            # 원본 이미지와 주석 처리된 이미지 나란히 표시
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_bytes, caption="원본 이미지", use_container_width=True)
            with col2:
                st.image(annotated_image, caption="얼굴 분석 결과", use_container_width=True)

            st.subheader("👤 상세 얼굴 분석 결과")
            
            for idx, face in enumerate(face_details):
                st.markdown(f"### 🟢 얼굴 {idx+1} 분석 결과")
                
                # 박스 색상과 동일한 색상의 구분선 추가
                box_colors = ["red", "blue", "green", "yellow", "purple"]
                box_color = box_colors[idx % len(box_colors)]
                st.markdown(f"<hr style='border: 2px solid {box_color}'>", unsafe_allow_html=True)
                
                age_range = face['AgeRange']
                gender = face['Gender']
                emotions = face['Emotions']
                top_emotion = max(emotions, key=lambda x: x['Confidence'])
                
                # 기본 정보 표시
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("성별", f"{gender['Value']} ({gender['Confidence']:.1f}%)")
                with col2:
                    st.metric("추정 나이", f"{age_range['Low']}~{age_range['High']}세")
                with col3:
                    st.metric("주요 감정", f"{top_emotion['Type']} ({top_emotion['Confidence']:.1f}%)")
                
                # 감정 차트
                st.write("**감정 분석 결과:**")
                emotions_sorted = sorted(emotions, key=lambda x: -x['Confidence'])
                for emotion in emotions_sorted:
                    st.progress(int(emotion['Confidence']), 
                              text=f"{emotion['Type']}: {emotion['Confidence']:.1f}%")
                
                # 추가 얼굴 속성
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

    # 이미지 업로드
    st.subheader("📷 비교할 두 장의 얼굴 사진을 업로드하세요")
    col1, col2 = st.columns(2)
    with col1:
        image1 = st.file_uploader("이미지 1", type=["jpg", "jpeg", "png"], key="img1")
    with col2:
        image2 = st.file_uploader("이미지 2", type=["jpg", "jpeg", "png"], key="img2")

    if image1 and image2:
        image1_bytes = image1.read()
        image2_bytes = image2.read()

        # Rekognition 얼굴 비교 요청
        try:
            response = client.compare_faces(
                SourceImage={'Bytes': image1_bytes},
                TargetImage={'Bytes': image2_bytes},
                SimilarityThreshold=50
            )

            # 결과 표시
            st.subheader("🔍 비교 결과")
            
            # 이미지 나란히 표시
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
                
                # 유사도 게이지로 표시
                st.write(f"**얼굴 유사도:** {similarity:.2f}%")
                st.progress(int(similarity), text=f"유사도 {similarity:.2f}%")
                
                # 유사도 기반 메시지
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