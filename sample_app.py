import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.figure import Figure

#解析##########################################################################################################################
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from tqdm import tqdm
import datetime as dt
from scipy import interpolate #補完


# ページ情報、基本的なレイアウト
st.set_page_config(
    page_title="北海道リモセンPJ",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("北海道リモートセンシングPJ 🧅")
st.markdown("""
リモートセンシングによる玉ねぎ圃場のNDVIモニタリングおよび施肥管理を検討します。
""")

### 【リンク】
#- [つくば分析センター](https://www.katakuraco-op.com/site_tsukuba/)： こちらで土壌の受託分析を行っています。

# ファイルアップロードして表示
data = st.file_uploader("file_upload", type="csv") 

if data:
        dat =  pd.read_csv(data, index_col=0, encoding = "shift-jis")
        dat2 =  dat
        data_disp = st.checkbox('データベース表示')
        if data_disp == True :
            st.dataframe(dat2,height=200,width=2000)
        
        # サイドバー
        st.sidebar.title("NDVI解析条件")
        dev_key = st.sidebar.selectbox("キー", ("統括ID","生産圃"))
        ind_dev_key_stocks = st.sidebar.multiselect(label="対象ID",options=dat2[dev_key].unique())
        
        if ind_dev_key_stocks:
                min_date = dt.date(2023, 4, 1)
                max_date = dt.date(2023, 9, 30)
                time_course = st.sidebar.slider('調査期間を指定してください。', value=(min_date, max_date), min_value=min_date, max_value=max_date)
                ndvi_pre_cutoff = st.sidebar.slider("NDVI_pre_cutoff", min_value=0.00, max_value=1.00,value=0.10, step=0.05)
                ndvi_cutoff = st.sidebar.slider("NDVI_growth_cutoff", min_value=0.10, max_value=1.00,value=0.50, step=0.05)
                hampel_MAD_cutoff = st.sidebar.slider("hampel_MAD_cutoff", min_value=0.0, max_value=10.0,value=1.0, step=0.5)
                hampel_window_size = st.sidebar.slider("hampel_window_size", min_value=1, max_value=100,value=2, step=1)
                sg_window_size  = st.sidebar.slider("sg_window_size", min_value=3, max_value=101,value=15, step=2)
                sg_polyorder = st.sidebar.slider("sg_polyoder", min_value=1, max_value=20,value=10, step=1)    #sgの多項式近似
                spline_k = st.sidebar.slider("spline_k", min_value=1, max_value=5,value=3, step=1)    #スプラインの多項式、デフォルト3
                spline_s = st.sidebar.slider("spline_s", min_value=0.00, max_value=1.00,value=0.01, step=0.01)    #スプラインのs
                
                # input_num =st.sidebar.number_input('強さ：',0,100,0)
                # input_text =st.sidebar.text_input('国を入力', 'Japan')
                # select_num =st.sidebar.number_input('年(1952~5年おき)',1952,2007,1952,step=5)
                
                #hampel filterを定義#####################################################################################
                import numpy as np
                # '''
                # * Input
                #     * x       input data
                #     * k       half window size (full 2*k+1)          
                #     * thr     threshold (defaut 3), optional
                 
                # * Output
                #     * output_x    filtered data
                #     * output_Idx indices of outliers
                # '''
                def Hampel(x, k, thr=3):
                    arraySize = len(x)
                    idx = np.arange(arraySize)
                    output_x = x.copy()
                    output_Idx = np.zeros_like(x)
                 
                    for i in range(arraySize):
                        mask1 = np.where( idx >= (idx[i] - k) ,True, False)
                        mask2 = np.where( idx <= (idx[i] + k) ,True, False)
                        kernel = np.logical_and(mask1, mask2)
                        median = np.median(x[kernel])
                        std = 1.4826 * np.median(np.abs(x[kernel] - median))
                
                        if np.abs(x[i] - median) > thr * std:
                            output_Idx[i] = 1
                            output_x[i] = median
                 
                    # return output_x, output_Idx.astype(bool)
                    return output_x, output_Idx
                
                
                
                #データベースファイル(csv)を規定########################################################################################
                dat["Date"] = pd.to_datetime(dat["Date"]) #Dateを日付型で定義
                dat =dat.query('Date >= @time_course[0] and Date <= @time_course[1]')
                
                #処理条件を設定########################################################################################################
                dtk = "PB"  #利用するNDVI_meanのDataTypeキー。                       #★★★★★★★★★★★★
                dat_dtk = dat[dat.DataType == dtk] #DataTypeがtarget_keyの行のみ抽出。ほかにPTとかもある。
                dat_dtk = dat_dtk[dat_dtk.NDVI_mean >= ndvi_pre_cutoff] 
                
                #グラフ用リスト作成####################################################################################################
                r=list()
                c=list()
                dev_by = ind_dev_key_stocks  #dat_dtk[dev_key].unique() #分割基準とするカラム名
                
                for j in range(1,abs(len(dev_by)//2)+2): 
                    for jj in[1,2]:
                        c.append(jj) #行リスト作成       
                jjj=[1,1]
                for j in range(1,abs(len(dev_by)//2)+2):
                    for jj in[j,j]:
                        r.append(jj) #列リスト作成
                        jjj=jjj+[1,1]
                        
                #グラフ描写###########################################################################################################
                whole_res = pd.DataFrame()
                whole_hokan = pd.DataFrame()
                fig = make_subplots(rows=max(r), cols=max(c),subplot_titles=dev_by)
                for i,n in enumerate(tqdm(dev_by)):
                    
                    #Hampel filerで異常値検出##########################################################################################
                    sg=pd.DataFrame()
                    sg = Hampel(dat_dtk[dat_dtk[dev_key] == n]["NDVI_mean"].to_numpy(), hampel_window_size, thr=hampel_MAD_cutoff)[0]
                    sg_err = pd.DataFrame(Hampel(dat_dtk[dat_dtk[dev_key] == n]["NDVI_mean"].to_numpy(), hampel_window_size, thr=hampel_MAD_cutoff)[1]*dat_dtk[dat_dtk[dev_key] == n]["NDVI_mean"])
                    sg_err["Date"] = dat_dtk[dat_dtk[dev_key] == n]["Date"].tolist()
                    sg_err["NDVI_mean"] = sg_err["NDVI_mean"].replace(0,"NA")
                    
                    #異常値処理後のポイントから平滑化トレンドラインを作る#################################################################
                    temp_sg = savgol_filter(sg, #元データ
                                       sg_window_size, #window数(正の奇数の整数で指定）
                                       polyorder = sg_polyorder, #多項式の次数
                                       deriv = 0)
                    
                    #解析条件、結果の集計###############################################################################################
                    res = pd.DataFrame()
                    res[dev_key]= [dev_key]*len(sg)
                    res["dev_name"]= [n]*len(sg)
                    res["type_key"] = [dtk]*len(sg)
                    res["Date"] = dat_dtk[dat_dtk[dev_key] == n]["Date"].tolist()
                    res["ndvi_pre_cutoff"] = [ndvi_pre_cutoff]*len(sg)
                    res["raw"] =  dat_dtk[dat_dtk[dev_key] == n]["NDVI_mean"].tolist()
                    res["trend"] = temp_sg
                    res["outlier"] = Hampel(dat_dtk[dat_dtk[dev_key] == n]["NDVI_mean"].to_numpy(), hampel_window_size, thr=hampel_MAD_cutoff)[1]
                    res["higher"] = np.where(res['raw']>res["trend"], res['raw'],res["trend"])
                    res["raw-trend"] = (res['raw'] >= res["trend"]) #rawのほうがtrendによるトレンドよりも大きい　→outlier認定されていてもrawを採用する
                    res["raw-trend"] = [1 if q ==False else 0 for q in res["raw-trend"]]
                    res["outlier"] = res["outlier"] * res["raw-trend"]
                    res = res.drop("raw-trend", axis=1)
                    res["NDVI_cutoff"] =[ndvi_cutoff] *len(sg)
                    res["hampel_MAD_cutoff"] =[hampel_MAD_cutoff] *len(sg)
                    res["hampel_window_size"] =[hampel_window_size] *len(sg)
                    res["sg_window"] =[sg_window_size]*len(sg)
                    res["sg_polyorder"] =[sg_polyorder]*len(sg)
                
                    #補間行列を作る
                    hokan_res = res.loc[res.groupby('Date')['higher'].idxmax()].sort_index() #同日に2点以上のプロットがある場合は、最大値を残す
                    x_observed = pd.to_datetime(hokan_res["Date"]).values.astype(float) #一度float形に変換する
                    y_observed = hokan_res["higher"].tolist()
                    new_x = pd.to_datetime(pd.date_range(start=pd.to_datetime(x_observed.min()), end=pd.to_datetime(x_observed.max()), freq='D')).values.astype(float) #float形で連続xを作る
                    fitted_curve = interpolate.UnivariateSpline(x_observed, y_observed,w=hokan_res["higher"],k=spline_k,s=spline_s) #5次曲線で補完、ただし値が高いほど重みを大きくする
                    
                    hokan = pd.DataFrame()
                    hokan["dev_name"]= [n]*len(new_x)
                    hokan["Date"] = pd.to_datetime(new_x)
                    hokan["predict"] = fitted_curve(new_x)   
                        
                    #トレンドラインを描写（黒）#############################################################################
                    fig.add_trace(go.Scatter(x=hokan["Date"].tolist(),
                                  y=hokan["predict"].tolist(),      #高NDVIも低NDVIも異常値とみなす場合
                                  mode='lines',
                                  opacity=1,
                                  showlegend=False,
                                  visible=True,
                                  marker_color ="gray",
                                  line_width=3,
                                  #line_dash ="dot",
                                  name='trend_line',
                                  xaxis="x"+str(i+1),yaxis="y"+str(i+1)),
                                  row=r[i], col=c[i])
                
                    #異常値を描写（灰）#######################################################################################
                    fig.add_trace(go.Scatter(x=res[res["outlier"] != 0]["Date"].tolist(),
                                  y=res[res["outlier"] != 0]["raw"].tolist(),
                                  mode='markers',
                                  opacity=0.2,
                                  showlegend=False,              
                                  visible=True,
                                  marker_symbol= 'x',
                                  marker_color="black", 
                                  name="outlier",
                                  xaxis="x"+str(i+1),yaxis="y"+str(i+1)),
                                  row=r[i], col=c[i])  
                    
                    #≧ndvi_cutoffとなる期間を描写（赤）#############################################################################
                    fig.add_trace(go.Scatter(x=hokan[hokan["predict"] >= ndvi_cutoff]['Date'].tolist(),
                                  y=[1]*len(hokan[hokan["predict"] >= ndvi_cutoff]['Date']),      #高NDVIも低NDVIも異常値とみなす場合
                                  mode='lines',
                                  opacity=1,
                                  showlegend=False,
                                  visible=True,
                                  marker_color ="#d91e1e",
                                  line_width=10,
                                  name="NDVI >="+ str(ndvi_cutoff),
                                  xaxis="x"+str(i+1),yaxis="y"+str(i+1)),
                                  row=r[i], col=c[i])
                    
                    #正常値を描写（カラー）#######################################################################################
                    fig.add_trace(go.Scatter(x=res[res["outlier"] != 1]['Date'].tolist(),
                                  y=res[res["outlier"] != 1]["raw"].tolist(),
                                  mode='markers',
                                  opacity=1,
                                  showlegend=False,              
                                  visible=True,
                                  marker=dict(
                                        symbol="circle",
                                        size=10,
                                        color=res[res["outlier"] != 1]["raw"].tolist(),
                                        colorscale='portland', #https://www.self-study-blog.com/dokugaku/python-plotly-color-sequence-scales/
                                        cmin=0,  # Set the color scale limits
                                        cmax=ndvi_cutoff+0.1,
                                        showscale=False,
                                        line_width=1
                                    ),
                                  name="raw",
                                  xaxis="x"+str(i+1),yaxis="y"+str(i+1)),
                                  row=r[i], col=c[i])   
                    
                    #≧ndvi_cutoffとなる開始日、終了日、日数を表示###################################################################
                    temp_x=[]
                    temp_x2=[]
                
                    if len(list(hokan[hokan["predict"] >= ndvi_cutoff]['Date']))>0:
                        temp_x = list(hokan[hokan["predict"] >= ndvi_cutoff]['Date'])[0]
                        temp_x2 = list(hokan[hokan["predict"] >= ndvi_cutoff]['Date'])[len(list(hokan[hokan["predict"] >= ndvi_cutoff]['Date']))-1]
                        
                        #NDVI>=cutoffの期間を描写する
                        fig.add_annotation(x=temp_x,y=1,text= temp_x.strftime('%m/%d'),xref="x"+str(i+1),yref="y"+str(i+1),showarrow=True,font_color='black')
                        fig.add_annotation(x=temp_x2,y=1,text= temp_x2.strftime('%m/%d'),xref="x"+str(i+1),yref="y"+str(i+1),showarrow=True,font_color='black')
                      
                        #NDVI>=cutoffの期間を記録する
                        hokan["s_NDVI≧cutoff"] = [temp_x]*len(hokan["Date"])
                        hokan["e_NDVI≧cutoff"] = [temp_x2]*len(hokan["Date"])
                        d_format = '%Y/%m/%d'
                        s_to_e_days = temp_x2- temp_x
                        hokan["days_NDVI≧cutoff"] = [s_to_e_days.days]*len(hokan["Date"])
                        fig.add_annotation(x=temp_x2,y=0.1,text= str(s_to_e_days.days)+"days",xref="x"+str(i+1),yref="y"+str(i+1),showarrow=False,axref="x1", ayref="y1",font_size=12,font_color='black')
                
                    else:
                        hokan["s_NDVI≧cutoff"] = [0]*len(hokan["Date"])
                        hokan["e_NDVI≧cutoff"] = [0]*len(hokan["Date"])  
                        hokan["days_NDVI≧cutoff"] = [0]*len(hokan["Date"])
                
                    #データを保存する#################################################################################################
                    whole_res = pd.concat([whole_res, res], axis=0)
                    hokan = pd.merge(res, hokan, how='outer',on="Date")
                    whole_hokan = pd.concat([whole_hokan, hokan], axis=0)
                
                # 2列で表示、8分割して1が5つ分、2が3つ分
                col1, col2 = st.columns([5,3])
                
                if len(dev_by)<=1:
                    fig.update_layout(yaxis_title = 'NDVI_mean',autosize=False, width=1000,height=330,plot_bgcolor="whitesmoke")
                else:
                    fig.update_layout(yaxis_title = 'NDVI_mean',autosize=False, width=1000,height=300+300*(len(dev_by)//2),plot_bgcolor="whitesmoke")
                fig.update_xaxes(linecolor='lightgray', gridcolor='lightgray',mirror=True,tickformat="%Y-%m-%d",tickangle=-30,dtick ='M1')
                fig.update_yaxes(range=(0,1.25),linecolor='lightgray', gridcolor='lightgray',mirror=True,dtick =0.25)
                
                if len(dev_by)>=1:
                    st_matrix = whole_hokan.drop_duplicates(subset=["dev_name_y"]).filter(items=["dev_name_y","s_NDVI≧cutoff","e_NDVI≧cutoff","days_NDVI≧cutoff"])
                    st_matrix.columns = [dev_key,"NDVI開始日","NDVI終了日","NDVI継続期間"]                                                           
                    st_matrix['NDVI開始日'] = pd.to_datetime(st_matrix['NDVI開始日']).dt.strftime('%Y-%m-%d')
                    st_matrix['NDVI終了日'] = pd.to_datetime(st_matrix['NDVI終了日']).dt.strftime('%Y-%m-%d')
                    st_matrix['NDVI継続期間'] = st_matrix['NDVI継続期間']
                
                
                #グラフ数が少ない場合のみ画面出力する
                if i <=100:
                    col1.title("NDVIの推移")
                    col1.plotly_chart(fig, use_container_width=True)
                    col2.title("解析データ")
                    col2.dataframe(st_matrix)
                    #st.plotly_chart(fig)
                
                #ダウンロードボタン
                import base64
                csv = whole_hokan.fillna("NaN").sort_values(['dev_name_y', 'Date']).to_csv(sep=",")
                b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="result_utf-8-sig.csv">Download Link</a>'
                st.markdown(f"CSVファイルのダウンロード:  {href}", unsafe_allow_html=True)
                
                #解析を出力する########################################################################################################
                #fig.write_html(R"C:\Users\220127\Desktop\SpaceAgri_Download_Data_2023.html") 
                #whole_res.to_csv(R"C:\Users\220127\Desktop\SpaceAgri_Download_Data_2023_res.csv",encoding="shift-jis",sep=",")
                #whole_hokan.fillna("NaN").sort_values(['dev_name_y', 'Date']).to_csv(R"C:\Users\220127\Desktop\SpaceAgri_Download_Data_2023_hokan2.csv",encoding="shift-jis",sep=",")
                #print("fin.")
        
