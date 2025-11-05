 import pandas as pd
 from scipy import stats
 import numpy as np
 import matplotlib.pyplot as plt
 import webbrowser
 import seaborn as sns
 from scipy.stats import trim_mean
 import folium
 import geokakao as gk
 import plotly.express as px
 import matplotlib.image as mpimg
 import plotly.graph_objects as go

 # 한글
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

 # 데이터셋 로드
Usage = pd.read_csv('C:/Users/rlb04/OneDrive/바탕 화면/김가람/2학년 2학기/데이터사이언스/기말고사/서울특별시_공공와이파이 AP별 사용량_10_13_2021/서울특별시 공공와이파이 AP별 사용량_고정형(20210501_20211013).csv', encoding = 'cp949')
Location = pd.read_csv('C:/Users/rlb04/OneDrive/바탕 화면/김가람/2학년 2학기/데이터사이언스/기말고사/서울시 공공와이파이 서비스 위치 정보.csv', encoding = 'cp949')
Population = pd.read_csv('C:/Users/rlb04/OneDrive/바탕 화면/김가람/2학년 2학기/데이터사이언스/기말고사/인구밀도.csv')
Time = pd.read_csv('C:/Users/rlb04/OneDrive/바탕 화면/김가람/2학년 2학기/데이터사이언스/기말고사/인터넷 이용시간.csv')

# 1. 공공 Wi-Fi 설치 현황과 이용 수요의 관계 분석
# 1) Wi-Fi 설치 수
# 데이터 전처리
# 특이값 (과천시) 제거
Location = Location[Location['자치구'] != '과천시']
Location

# 연도벌
install_count_by_year = Location.groupby('설치년도').size()
install_count_by_year

# 지역별(구) - 가나다순 정렬
install_count_by_region = Location['자치구'].value_counts().sort_index()
install_count_by_region

# 총합
total_install_count = Location.shape[0]
total_install_count

# 전체 설치 수에서 각 지역의 비율
install_percentage_by_region = install_count_by_region / total_install_count * 100
install_percentage_by_region

# 시각화 : 설치 연도별 현황 (선 그래프)
plt.figure(figsize = (10, 6))
plt.plot(install_count_by_year.index, install_count_by_year.values, marker = 'o', color = 'b', label = '설치 수')
plt.title('연도별 공공 Wi-Fi 설치 수')
plt.xlabel('설치년도')
plt.ylabel('설치 수')
plt.grid(True)
plt.legend()
plt.show()

# 시각화 : 지역별 설치 현황 (막대 그래프)
plt.figure(figsize = (12, 8))
sns.barplot(x = install_count_by_region.index, y = install_count_by_region.values, palette = 'viridis')
plt.title('지역별 공공 Wi-Fi 설치 수')
plt.xlabel('자치구')
plt.ylabel('설치 수')
plt.xticks(rotation = 45)
plt.show()

# 시각화 : 지역별 설치 비율 (원 그래프)
plt.figure(figsize = (8, 8))
plt.pie(install_percentage_by_region, labels = install_percentage_by_region.index, autopct = '%1.1f%%', startangle = 140, colors = sns.color_palette('pastel'))
plt.title('지역별 공공 Wi-Fi 설치 비율')
plt.show()

# 2) Wi-Fi 사용량
# 데이터 전처리
# 특이값 (과천시) 제거
Usage = Usage[Usage['자치구'] != '과천시']

# 지역별 사용량 통계 계산 - 가나다순 정렬
usage_stats = Usage.groupby('자치구')['AP별 이용량(GB)'].agg(
    평균 = 'mean',중앙값 = 'median',
    분산 = 'var',표준편차 = 'std',
    절사평균 = lambda x: trim_mean(x, 0.1)
).sort_index()
usage_stats

# 시각화 : 지역별 사용량 평균 (막대 그래프)
plt.figure(figsize = (12, 8))
usage_stats['평균'].plot(kind = 'bar', color = 'skyblue', alpha = 0.8, edgecolor = 'black')
plt.title('지역별 Wi-Fi 사용량 평균')
plt.xlabel('자치구')
plt.ylabel('평균 사용량 (GB)')
plt.xticks(rotation = 45)
plt.show()

# 시각화 : 지역별 사용량 분포 (상자 그림)
plt.figure(figsize = (12, 8))
sns.boxplot(x = '자치구', y = 'AP별 이용량(GB)', data = Usage, palette = 'coolwarm')
plt.title('지역별 Wi-Fi 사용량 분포')
plt.xlabel('자치구')
plt.ylabel('Wi-Fi 사용량 (GB)')
plt.xticks(rotation = 45)
plt.show()

# 3) Wi-Fi 설치 수와 사용량의 관계
# 데이터 전처리
# 두 데이터셋에 모두 존재하는 Wi-Fi만 추출
common_ids = set(Location['관리번호']) & set(Usage['관리번호'])
common_ids

filtered_location_data = Location[Location['관리번호'].isin(common_ids)]
filtered_usage_data = Usage[Usage['관리번호'].isin(common_ids)]

# 가나다순 정렬
filtered_location_data = filtered_location_data.sort_values(by = '자치구')
filtered_usage_data = filtered_usage_data.sort_values(by = '자치구')
filtered_location_data
filtered_usage_data

# 지역별 Wi-Fi 설치 수
install_counts = filtered_location_data.groupby('자치구').size().reset_index(name = '설치 수')
install_counts

# 지역별 설치 수가 다르므로 사용량은 평균을 내어 사용
usage_mean = filtered_usage_data.groupby('자치구')['AP별 이용량(GB)'].mean().reset_index(name = '평균 사용량')
usage_mean

# 두 변수 병합
cor_data = pd.merge(install_counts, usage_mean, on = '자치구').dropna()
cor_data

# 상관계수 계산
cor = cor_data['설치 수'].corr(cor_data['평균 사용량'])
cor

# 시각화 : Wi-Fi 설치 수와 사용량 합계의 상관관계
plt.figure(figsize = (10, 6))
sns.regplot(
    x = '설치 수',
    y = '평균 사용량',
    data = cor_data,
    scatter_kws = {'color' : 'blue', 's' : 50},
    line_kws = {'color' : 'red'}
)

plt.title(f'Wi-Fi 설치 수와 평균 사용량의 상관관계: {cor:.2f}', fontsize = 16)
plt.xlabel('Wi-Fi 설치 수')
plt.ylabel('Wi-Fi 평균 사용량 (GB)')
plt.grid(True)
plt.show()

 # 4) Wi-Fi 설치 위치 및 사용량 지도
# 데이터 병합
merged_data = pd.merge(filtered_location_data, filtered_usage_data, on = '관리번호', how = 'inner')
merged_data = merged_data.rename(columns = {'자치구_x' : '자치구'})
merged_data = merged_data[['자치구', '관리번호', '와이파이명', '도로명주소', '설치년도', 'X좌표', 'Y좌표', 'AP별 이용량(GB)']].dropna()

# 도로명주소를 주소로 변경 ex) 서소문로 51 -> 서울 서대문구 서소문로 51
merged_data['도로명주소'] = '서울' + ' ' + merged_data['자치구'] + ' ' + merged_data['도로명주소']
merged_data = merged_data.rename(columns = {'도로명주소' : '주소'})
merged_data

# 서울 중심 좌표 (기준점)
center = [37.5665, 126.9780]

# Folium 지도 생성
map = folium.Map(location = center, zoom_start = 11)

# 설치 위치
for _, row in merged_data.iterrows():
    folium.Marker(location = (row['Y좌표'], row['X좌표']),
        icon = folium.Icon(color = 'blue', icon = 'signal')).add_to(map)

map.save('map.html')
webbrowser.open('map.html')
map = folium.Map(location = center, zoom_start = 12)

for _, row in merged_data.iterrows():
    folium.CircleMarker(
        location = (row['Y좌표'], row['X좌표']),
        radius = row['AP별 이용량(GB)'] / 400,
        color = 'blue', fill = True,
        fill_opacity = '80%',
        tooltip = f'{row['와이파이명']}<br>사용량: {row['AP별 이용량(GB)']:.2f} GB'
    ).add_to(map)

map.save('map.html')
webbrowser.open('map.html')

# 2. 인터넷 사용 시간과 공공 Wi-Fi 사용량의 관계 분석
# 1) 인터넷 사용 시간
filtered_time = Time[['시점', '주 평균(시간)']].rename(columns = {'시점' : '연도', '주 평균(시간)' : '주 평균 이용 
시간(시간)'})
 filtered_time = filtered_time[filtered_time['연도'].str.isnumeric()]
 filtered_time['연도'] = pd.to_numeric(filtered_time['연도'], errors = 'coerce')
 filtered_time['주 평균 이용 시간(시간)'] = pd.to_numeric(filtered_time['주 평균 이용 시간(시간)'], errors = 'coerce')
 filtered_time = filtered_time.dropna()
 filtered_time

 # 시각화 : 연도별 인터넷 사용 시간 변화 (선 그래프)
 plt.figure(figsize = (10, 6))

 plt.plot(
    filtered_time['연도'], filtered_time['주 평균 이용 시간(시간)'],
    marker = 'o', linestyle = '-',
    color = 'blue', label = '주 평균 이용 시간'

)
 plt.title('연도별 인터넷 사용 시간 변화', fontsize = 16)
 plt.xlabel('연도', fontsize = 14)
 plt.ylabel('주 평균 이용 시간 (시간)', fontsize = 14)
 plt.grid(True)
 plt.legend(fontsize = 12)
 plt.show()

# 2) Wi-Fi 설치 수와 인터넷 사용 시간의 관계
# 연도별 Wi-Fi 누적 설치 수
wifi_year_total = Location[['설치년도']].copy()
wifi_year_total['설치년도'] = pd.to_numeric(wifi_year_total['설치년도'], errors = 'coerce')
wifi_year_total = wifi_year_total.groupby('설치년도').size().cumsum().reset_index()
wifi_year_total.columns = ['연도', '누적 설치 수']
wifi_year_total = wifi_year_total.dropna()
wifi_year_total

# 병합
combined_data = pd.merge(filtered_time, wifi_year_total, on='연도', how='inner')
combined_data

# 상관분석
correlation = combined_data['주 평균 이용 시간(시간)'].corr(combined_data['누적 설치 수'])
correlation

# 시각화
combined_data.plot.scatter(x = '주 평균 이용 시간(시간)', y = '누적 설치 수')
m, b = np.polyfit(combined_data['주 평균 이용 시간(시간)'], combined_data['누적 설치 수'], 1)
plt.plot(combined_data['주 평균 이용 시간(시간)'], m * np.array(combined_data['주 평균 이용 시간(시간)']) + b)
plt.title(f'Wi-Fi 누척 설치 수와 인터넷 사용 시간의 상관관계: {correlation:.2f}', fontsize = 16)
plt.xlabel('주 평균 인터넷 사용 시간 (시간)')
plt.ylabel('Wi-Fi 누적 설치 수')
plt.grid(True)
plt.show()

# 3. 공공 Wi-Fi 설치와 지역별 인구 밀도의 관계 분석
# 1) 인구 밀도와 설치 수의 관계
# 설치 수 데이터 (설치년도 2023년까지만)
install_data = merged_data[merged_data['설치년도'] <= 2023]
install_data = install_data.groupby('자치구').size().reset_index(name = '설치 수')
install_data

# 인구 밀도 데이터
density_data = Population.drop(columns = ['동별(1)'])
density_data = density_data.rename(columns = {'동별(2)' : '자치구'})
density_data = density_data.iloc[1:]
density_data

# 2023년 인구 밀도만 추출
density_2023 = density_data[['자치구', '2023']]
density_2023 = density_2023.rename(columns = {'2023' : '인구 밀도'})
density_2023

 # 병합
merged_density = pd.merge(density_2023, install_data, on = '자치구', how = 'inner')
merged_density['인구 밀도'] = pd.to_numeric(merged_density['인구 밀도'], errors='coerce')
merged_density['설치 수'] = pd.to_numeric(merged_density['설치 수'], errors='coerce')
merged_density

# 상관분석
cor_density = merged_density[['인구 밀도', '설치 수']].corr().iloc[0, 1]
cor_density

# 시각화 : 인구 밀도와 Wi-Fi 설치 수의 관계
merged_density.plot.scatter(x = '인구 밀도', y = '설치 수')
m, b = np.polyfit(merged_density['인구 밀도'], merged_density['설치 수'], 1)
plt.plot(merged_density['인구 밀도'], m * np.array(merged_density['인구 밀도']) + b)
plt.title(f'인구 밀도와 공공 Wi-Fi 설치 수의 상관관계: {cor_density:.2f}', fontsize = 16)
plt.xlabel('인구 밀도')
plt.ylabel('Wi-Fi 설치 수')
plt.grid(True)
plt.show()

# 2) 인구 밀도와 사용량의 관계
# 인구 밀도와 사용량의 데이터 분포와 변동성 분석
# 사분위수, 사분범위, 표준편차
# 2021년 인구 밀도만 (Wi-Fi 사용량 통계가 2021년 기준임)
density_2021 = density_data[['자치구', '2021']]
density_2021 = density_2023.rename(columns = {'2021' : '인구 밀도'})
density_2021

usage_data = merged_data
usage_data = usage_data.groupby('자치구')['AP별 이용량(GB)'].mean().reset_index(name = '평균 사용량')
usage_data

# 병합
density_usage_data = pd.merge(density_2021, usage_data, on = '자치구', how = 'inner')
density_usage_data['인구 밀도'] = pd.to_numeric(density_usage_data['인구 밀도'], errors='coerce')
density_usage_data['평균 사용량'] = pd.to_numeric(density_usage_data['평균 사용량'], errors='coerce')
density_usage_data

# 분석
quartiles = density_usage_data['평균 사용량'].quantile([0.25, 0.5, 0.75])
std_dev = density_usage_data['평균 사용량'].std()
quartiles
std_dev

# 시각화 : Wi-Fi 사용량 분포 (상자 그림)
plt.figure(figsize = (10, 6))
sns.boxplot(x = '평균 사용량',
data = density_usage_data,
palette = 'coolwarm')
plt.title('Wi-Fi 사용량 분포', fontsize = 16)
plt.xlabel('평균 Wi-Fi 사용량 (GB)', fontsize = 14)
plt.grid(True)
plt.show()

# 고밀도 지역과 저밀도 지역 간 사용량 차이 분석
median_density = density_usage_data['인구 밀도'].median()
high_density = density_usage_data[density_usage_data['인구 밀도'] >= median_density].reset_index()
low_density = density_usage_data[density_usage_data['인구 밀도'] < median_density].reset_index()

# 1. 독립 검정
# 고밀도 지역과 저밀도 지역은 다른 지역들이므로 독립된 표본임
# 2. 정규성 검정
pvalue = stats.shapiro(high_density['평균 사용량']).pvalue
pvalue

# pvalue = 0.28로 0.05 이상 -> 정규성 만족
pvalue = stats.shapiro(low_density['평균 사용량']).pvalue
pvalue

# pvalue = 0.40으로 0.05 이상 -> 정규성 만족
# 독립표본 T-검정 사용 가능
# 3. 등분산성 검정
pvalue = stats.levene(high_density['평균 사용량'], low_density['평균 사용량']).pvalue
pvalue

# pvalue = 0.22로 0.05 이상 -> equal_var = True
result = stats.ttest_ind(high_density['평균 사용량'], low_density['평균 사용량'], equal_var = True)
result

# 시각화 : Wi-Fi 사용량 분포 (트리맵)
treemap_data = density_usage_data[['자치구', '평균 사용량', '인구 밀도']]
fig = px.treemap(
    data_frame = treemap_data,
    path = ['자치구'], # 계층 구조
    values = '평균 사용량', # 타일 면적
    color = '인구 밀도', # 색상
    color_continuous_scale = 'Bluyl'
)

fig.update_layout(
    margin_t = 50,
    margin_l = 25,
    margin_r = 25,
    margin_b = 25,
    width = 800, 
    height = 600,
    title_text = '자치구별 인구 밀도와 평균 사용량',
    title_font_size = 20
)

fig.write_html('C:/Users/rlb04/OneDrive/바탕 화면/김가람/2학년 2학기/데이터사이언스/기말고사/treemap.html')
webbrowser.open('C:/Users/rlb04/OneDrive/바탕 화면/김가람/2학년 2학기/데이터사이언스/기말고사/treemap.html')

# 3) 인구 밀도, 설치 수, 사용량의 관계
# 설치 수
install_data

# 시각화 (버블 차트)
bubble_data = pd.merge(density_usage_data, install_data, on = '자치구', how = 'inner')

sns.scatterplot(
    data = bubble_data,
    x = '인구 밀도', y = '평균 사용량',
    size = '설치 수', sizes = (20, 4000),
    hue = '자치구', alpha = 0.5,
    legend = False
)

plt.xlim(bubble_data['인구 밀도'].min() * 0.9, bubble_data['인구 밀도'].max() * 1.1)
plt.ylim(bubble_data['평균 사용량'].min() * 0.9, bubble_data['평균 사용량'].max() * 1.1)

for i in range(bubble_data.shape[0]):
    plt.text(
    x = bubble_data['인구 밀도'].iloc[i],
    y = bubble_data['평균 사용량'].iloc[i],
    s = bubble_data['자치구'].iloc[i],
    horizontalalignment = 'center',
    size = 'small',
    color = 'dimgray'
    )

plt.title('인구 밀도, 설치 수, 사용량 간의 관계', fontsize = 16)
plt.xlabel('인구 밀도', fontsize = 14)
plt.ylabel('Wi-Fi 사용량 (GB)', fontsize = 14)
plt.grid(True)
plt.show()

