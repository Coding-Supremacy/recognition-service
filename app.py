import streamlit as st
import boto3

# Rekognition 클라이언트 생성
client = boto3.client('rekognition', region_name='ap-northeast-2')

# 사이드바 메뉴 설정
st.sidebar.title("메뉴 선택")
app_mode = st.sidebar.selectbox(
    "기능을 선택하세요",
    ["얼굴 감정 분석", "얼굴 비교"]
)

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

        st.image(image_bytes, caption="업로드한 이미지", use_container_width=True)
        st.subheader("👤 얼굴 분석 결과")

        face_details = response['FaceDetails']

        if len(face_details) == 0:
            st.warning("얼굴이 감지되지 않았습니다.")
        else:
            for idx, face in enumerate(face_details):
                st.markdown(f"### 얼굴 {idx+1}")
                age_range = face['AgeRange']
                gender = face['Gender']
                emotions = face['Emotions']

                # 📌 감정 중 Confidence가 가장 높은 것 추출
                top_emotion = max(emotions, key=lambda x: x['Confidence'])
                emotion_type = top_emotion['Type']
                emotion_confidence = top_emotion['Confidence']

                # 😊 감정에 따른 메시지 설정
                emotion_message = {
                    "HAPPY": "행복한 사진입니다 😊",
                    "SAD": "조금 슬퍼보이네요 😢",
                    "ANGRY": "화난 표정처럼 보여요 😠",
                    "CONFUSED": "조금 혼란스러워 보이네요 🤔",
                    "DISGUSTED": "불쾌한 표정이네요 😖",
                    "SURPRISED": "놀란 표정이네요 😲",
                    "CALM": "차분한 분위기예요 😌",
                    "FEAR": "두려워하는 것 같아요 😨"
                }.get(emotion_type, "감정을 파악하기 어려운 사진입니다.")

                # 출력
                st.write(f"- 나이 추정: **{age_range['Low']} ~ {age_range['High']}세**")
                st.write(f"- 성별 추정: **{gender['Value']}** ({gender['Confidence']:.2f}%)")
                st.write(f"- 주요 감정: **{emotion_type}** ({emotion_confidence:.2f}%)")
                st.success(emotion_message)

                st.divider()

elif app_mode == "얼굴 비교":
    st.title("🧑‍🤝‍🧑 얼굴 비교기 (Amazon Rekognition)")

    # 이미지 업로드
    st.subheader("📷 비교할 두 장의 얼굴 사진을 업로드하세요")
    image1 = st.file_uploader("이미지 1", type=["jpg", "jpeg", "png"], key="img1")
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

            st.subheader("🔍 업로드한 이미지")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image1_bytes, caption="이미지 1", use_container_width=True)
            with col2:
                st.image(image2_bytes, caption="이미지 2", use_container_width=True)

            st.subheader("🔎 비교 결과")

            face_matches = response['FaceMatches']
            if len(face_matches) == 0:
                st.warning("두 사람은 동일 인물로 인식되지 않았습니다.")
            else:
                match = face_matches[0]
                similarity = match['Similarity']
                st.write(f"✔️ 유사도: **{similarity:.2f}%**")

                # 유사도 기반 메시지
                if similarity > 95:
                    st.success("같은 사람일 가능성이 매우 높습니다 😊")
                elif similarity > 80:
                    st.info("같은 사람일 가능성이 있습니다 🙂")
                else:
                    st.warning("다른 사람일 가능성이 높습니다 😐")

        except Exception as e:
            st.error(f"에러 발생: {str(e)}")