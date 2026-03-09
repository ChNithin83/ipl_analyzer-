import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="IPL Analytics Hub", page_icon="🏏",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#060a12;color:#c9d1d9;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a0e1a,#111827);border-right:1px solid #ff6b0030;}
h1,h2,h3{font-family:'Rajdhani',sans-serif!important;letter-spacing:1px;}
.hero{font-family:'Rajdhani',sans-serif;font-size:clamp(2rem,5vw,3.5rem);font-weight:700;
      background:linear-gradient(120deg,#ff6b00,#ffd700,#ff6b00);background-size:200%;
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      animation:shimmer 3s linear infinite;letter-spacing:3px;}
@keyframes shimmer{0%{background-position:0%}100%{background-position:200%}}
.sub{color:#6e7e96;font-size:.9rem;letter-spacing:2px;text-transform:uppercase;margin-top:-6px;}
.kpi{background:linear-gradient(135deg,#0d1320,#1a2030);border:1px solid #ff6b0030;
     border-radius:14px;padding:18px 12px;text-align:center;
     box-shadow:0 4px 20px #ff6b0010;transition:transform .2s,border-color .2s;}
.kpi:hover{transform:translateY(-4px);border-color:#ff6b00;}
.kv{font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:#ff6b00;}
.kl{font-size:.72rem;color:#6e7e96;text-transform:uppercase;letter-spacing:1.2px;}
.sec{font-family:'Rajdhani',sans-serif;font-size:1.4rem;font-weight:700;color:#ffd700;
     border-bottom:2px solid #ff6b0035;padding-bottom:6px;margin:26px 0 14px;}
.pcard{background:linear-gradient(135deg,#0d1320,#1a2030);border:1px solid #ff6b0030;
       border-radius:16px;padding:26px;text-align:center;}
.badge{display:inline-block;padding:3px 11px;border-radius:20px;font-size:.7rem;
       font-weight:700;text-transform:uppercase;letter-spacing:1px;}
.pred{background:linear-gradient(135deg,#0a1a0a,#112011);border:2px solid #00d084;
      border-radius:18px;padding:30px;text-align:center;box-shadow:0 0 40px #00d08420;}
.pval{font-family:'Rajdhani',sans-serif;font-size:3.5rem;font-weight:700;color:#00d084;}
.lbrow{display:flex;justify-content:space-between;align-items:center;
       padding:10px 14px;border-radius:10px;margin-bottom:6px;
       background:#0d1320;border:1px solid #1c2333;transition:border-color .2s;}
.lbrow:hover{border-color:#ff6b0060;}
.lbrank{font-family:'Rajdhani',sans-serif;font-size:1.3rem;font-weight:700;
        color:#ff6b00;min-width:32px;}
.lbname{flex:1;padding:0 12px;font-weight:500;}
.lbval{color:#00d084;font-weight:700;font-family:'Rajdhani',sans-serif;font-size:1.1rem;}
</style>
""", unsafe_allow_html=True)

T = dict(paper_bgcolor="#060a12", plot_bgcolor="#060a12",
         font=dict(color="#c9d1d9", family="DM Sans"),
         xaxis=dict(gridcolor="#1c2333", linecolor="#21262d"),
         yaxis=dict(gridcolor="#1c2333", linecolor="#21262d"))
OR,GD,GR,BL = "#ff6b00","#ffd700","#00d084","#4f8ef7"

@st.cache_data
def load():
    try:
        df = pd.read_csv("ipl_data.csv")
        return df.dropna(subset=["team", "role", "player", "season"])
    except FileNotFoundError:
        st.error("❌ Run `python3 preprocess.py` first!")
        st.stop()

@st.cache_resource
def train(df):
    le_r=LabelEncoder(); le_t=LabelEncoder()
    le_c=LabelEncoder(); le_ti=LabelEncoder()
    X=df.copy()
    X["re"]=le_r.fit_transform(X["role"])
    X["te"]=le_t.fit_transform(X["team"])
    X["ce"]=le_c.fit_transform(X["country"])
    X["tie"]=le_ti.fit_transform(X["tier"])
    feats=["matches","runs","strike_rate","batting_avg","fifties","hundreds",
           "wickets","economy","bowling_avg","catches","re","te","ce","tie"]
    Xf=X[feats]; y=X["auction_value_cr"]
    Xtr,Xte,ytr,yte=train_test_split(Xf,y,test_size=0.2,random_state=42)
    mods={
        "🌲 Random Forest":     RandomForestRegressor(n_estimators=150,random_state=42),
        "⚡ Gradient Boosting": GradientBoostingRegressor(n_estimators=150,random_state=42),
        "📐 Linear Regression": LinearRegression(),
    }
    out={}
    for n,m in mods.items():
        m.fit(Xtr,ytr); p=m.predict(Xte)
        out[n]={"model":m,"mae":round(mean_absolute_error(yte,p),3),"r2":round(r2_score(yte,p),3)}
    return out,le_r,le_t,le_c,le_ti,feats

df=load()
mods,le_r,le_t,le_c,le_ti,FEATS=train(df)
BEST=max(mods,key=lambda k:mods[k]["r2"])

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center;padding:16px 0 8px">
  <span style="font-family:Rajdhani;font-size:2rem;font-weight:700;color:#ff6b00">🏏 IPL HUB</span><br>
  <span style="color:#6e7e96;font-size:.78rem">REAL DATA · 2008–2024</span>
</div>""", unsafe_allow_html=True)

PAGE=st.sidebar.radio("",["🏠 Overview","👤 Player","📊 Teams",
                           "🏆 Leaderboard","🤖 Predictor","🔬 ML Models"])
st.sidebar.markdown("---")
st.sidebar.markdown('<span style="color:#6e7e96;font-size:.8rem">FILTERS</span>',unsafe_allow_html=True)
sel_s=st.sidebar.multiselect("Seasons",sorted(df["season"].unique()),default=sorted(df["season"].unique()))
sel_r=st.sidebar.multiselect("Roles",df["role"].unique(),default=list(df["role"].unique()))
sel_t=st.sidebar.multiselect("Teams",sorted(df["team"].unique()),default=sorted(df["team"].unique()))
dff=df[df["season"].isin(sel_s)&df["role"].isin(sel_r)&df["team"].isin(sel_t)]

def kpi(v,l,c):
    c.markdown(f'<div class="kpi"><div class="kv">{v}</div><div class="kl">{l}</div></div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════
if PAGE=="🏠 Overview":
    st.markdown('<div class="hero">🏏 IPL ANALYTICS HUB</div>',unsafe_allow_html=True)
    st.markdown('<div class="sub">Real Data · 606 Players · 2008–2024 · Ball by Ball Stats</div>',unsafe_allow_html=True)
    st.markdown("---")

    c1,c2,c3,c4,c5=st.columns(5)
    kpi(dff["player"].nunique(),"Players",c1)
    kpi(dff["season"].nunique(),"Seasons",c2)
    kpi(dff["team"].nunique(),"Teams",c3)
    kpi(f"{int(dff['runs'].sum()):,}","Total Runs",c4)
    kpi(int(dff["wickets"].sum()),"Total Wickets",c5)

    st.markdown('<div class="sec">🏆 Top 10 Run Scorers (All Time)</div>',unsafe_allow_html=True)
    top=dff.groupby("player")["runs"].sum().nlargest(10).reset_index()
    fig=px.bar(top,x="runs",y="player",orientation="h",
               color="runs",color_continuous_scale=["#1c2333",OR,GD],text="runs")
    fig.update_traces(textfont_color="white",textposition="outside")
    fig.update_layout(**T,height=380,showlegend=False,coloraxis_showscale=False)
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="sec">Top 10 Wicket Takers</div>',unsafe_allow_html=True)
        tw=dff.groupby("player")["wickets"].sum().nlargest(10).reset_index()
        fig2=px.bar(tw,x="wickets",y="player",orientation="h",
                    color="wickets",color_continuous_scale=["#1c2333",BL,GR])
        fig2.update_layout(**T,height=360,showlegend=False,coloraxis_showscale=False)
        fig2.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig2,use_container_width=True)

    with c2:
        st.markdown('<div class="sec">Role Distribution</div>',unsafe_allow_html=True)
        rd=dff.groupby("role")["player"].nunique().reset_index()
        fig3=px.pie(rd,names="role",values="player",
                    color_discrete_sequence=[OR,GD,GR,BL],hole=0.45)
        fig3.update_layout(**T,height=360)
        fig3.update_traces(textfont_color="white")
        st.plotly_chart(fig3,use_container_width=True)

    st.markdown('<div class="sec">Runs Trend by Season</div>',unsafe_allow_html=True)
    tr=dff.groupby(["season","role"])["runs"].sum().reset_index()
    fig4=px.line(tr,x="season",y="runs",color="role",
                 color_discrete_sequence=[OR,GD,GR,BL],markers=True)
    fig4.update_layout(**T,height=320)
    st.plotly_chart(fig4,use_container_width=True)

    st.markdown('<div class="sec">Strike Rate vs Runs (Bubble = Wickets)</div>',unsafe_allow_html=True)
    agg=dff.groupby(["player","role"]).agg(
        runs=("runs","sum"),sr=("strike_rate","mean"),wkts=("wickets","sum")).reset_index()
    agg=agg[agg["runs"]>200]
    fig5=px.scatter(agg,x="sr",y="runs",size="wkts",color="role",hover_name="player",
                    size_max=40,color_discrete_sequence=[OR,GD,GR,BL])
    fig5.update_layout(**T,height=400)
    st.plotly_chart(fig5,use_container_width=True)

    st.markdown('<div class="sec">Runs Distribution (Treemap)</div>',unsafe_allow_html=True)
    tm = dff.groupby(["team", "player"])["runs"].sum().reset_index()
    tm = tm[tm["runs"] > 100] # filter very small values for cleaner map
    fig_tm = px.treemap(tm, path=[px.Constant("All Teams"), "team", "player"], values="runs",
                        color="team", color_discrete_sequence=[OR, GD, GR, BL, "#d2386c"])
    fig_tm.update_layout(**T, height=450, margin=dict(t=20, l=0, r=0, b=0))
    st.plotly_chart(fig_tm, use_container_width=True)

# ═══════════════════════════════════════════════════
# PLAYER
# ═══════════════════════════════════════════════════
elif PAGE=="👤 Player":
    st.markdown('<div class="hero">PLAYER ANALYSIS</div>',unsafe_allow_html=True)
    st.markdown("---")

    player=st.selectbox("Select Player",sorted(dff["player"].unique()))
    if not player:
        st.warning("No players found for the selected filters. Please adjust your sidebar settings.")
        st.stop()
    pd_=dff[dff["player"]==player].sort_values("season")
    info=pd_.iloc[-1]

    icon={"Batsman":"🏏","WK-Batsman":"🧤","All-Rounder":"⭐","Bowler":"⚡"}.get(info["role"],"🏏")
    tc={"S":GD,"A":OR,"B":BL}.get(info["tier"],OR)

    c1,c2=st.columns([1,3])
    with c1:
        st.markdown(f"""
        <div class="pcard">
          <div style="font-size:3rem">{icon}</div>
          <div style="font-family:Rajdhani;font-size:1.7rem;font-weight:700;color:#ffd700">{player}</div>
          <div style="color:#6e7e96;margin:4px 0">{info['team']} · {info['country']}</div>
          <span class="badge" style="background:{OR}20;color:{OR};border:1px solid {OR}50">{info['role']}</span>
          <span class="badge" style="background:{tc}20;color:{tc};border:1px solid {tc}50;margin-left:6px">Tier {info['tier']}</span>
          <hr style="border-color:#1c2333;margin:14px 0">
          <div style="color:#6e7e96;font-size:.78rem">SEASONS PLAYED</div>
          <div style="font-family:Rajdhani;font-size:1.6rem;color:#ff6b00">{pd_['season'].nunique()}</div>
        </div>""",unsafe_allow_html=True)

    with c2:
        c1,c2,c3,c4=st.columns(4)
        kpi(f"{int(pd_['runs'].sum()):,}","Career Runs",c1)
        kpi(int(pd_["wickets"].sum()),"Career Wickets",c2)
        kpi(round(pd_["strike_rate"].mean(),1),"Avg Strike Rate",c3)
        kpi(f"₹{round(pd_['auction_value_cr'].mean(),1)}Cr","Avg Value",c4)

        kpi(str(int(pd_["highest"].max())) + ("*" if pd_["highest"].max() == pd_["highest"].max() else ""), "High Score", c1) # using c1 again creates a second row
        kpi(f"{int(pd_['wickets'].max())} wkts (Best)", "Best Bowling", c2)

    st.markdown('<div class="sec">Season-wise Performance</div>',unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["🏏 Batting","⚡ Bowling","💰 Auction Value"])

    with t1:
        fig=go.Figure()
        fig.add_trace(go.Bar(x=pd_["season"],y=pd_["runs"],name="Runs",marker_color=OR))
        fig.add_trace(go.Scatter(x=pd_["season"],y=pd_["strike_rate"],name="Strike Rate",
                                  yaxis="y2",line=dict(color=GD,width=2),mode="lines+markers"))
        fig.update_layout(**T,height=360,
                          yaxis2=dict(overlaying="y",side="right",gridcolor="#1c2333",color=GD),
                          legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig,use_container_width=True)

        c1,c2,c3=st.columns(3)
        kpi(int(pd_["fifties"].sum()),"Total 50s",c1)
        kpi(int(pd_["hundreds"].sum()),"Total 100s",c2)
        kpi(int(pd_["highest"].max()),"Highest Score",c3)

    with t2:
        fig2=go.Figure()
        fig2.add_trace(go.Bar(x=pd_["season"],y=pd_["wickets"],name="Wickets",marker_color=BL))
        fig2.add_trace(go.Scatter(x=pd_["season"],y=pd_["economy"],name="Economy",
                                   yaxis="y2",line=dict(color=GR,width=2),mode="lines+markers"))
        fig2.update_layout(**T,height=360,
                           yaxis2=dict(overlaying="y",side="right",gridcolor="#1c2333",color=GR),
                           legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig2,use_container_width=True)

    with t3:
        fig3=px.area(pd_,x="season",y="auction_value_cr",color_discrete_sequence=[GR])
        fig3.update_traces(fill="tozeroy",fillcolor="rgba(0,208,132,0.1)",line_color=GR)
        fig3.update_layout(**T,height=300)
        st.plotly_chart(fig3,use_container_width=True)

    st.markdown('<div class="sec">Full Stats Table</div>',unsafe_allow_html=True)
    cols_show=["season","matches","runs","strike_rate","batting_avg","fifties","hundreds",
               "highest","wickets","economy","catches","auction_value_cr"]
    st.dataframe(pd_[cols_show].set_index("season").style.format("{:.1f}"),use_container_width=True)

# ═══════════════════════════════════════════════════
# TEAMS
# ═══════════════════════════════════════════════════
elif PAGE=="📊 Teams":
    st.markdown('<div class="hero">TEAM COMPARISON</div>',unsafe_allow_html=True)
    st.markdown("---")

    teams=sorted(dff["team"].unique())
    ca,cb=st.columns(2)
    ta=ca.selectbox("Team A",teams,index=0)
    tb=cb.selectbox("Team B",teams,index=min(1,len(teams)-1))
    da=dff[dff["team"]==ta]; db=dff[dff["team"]==tb]

    metrics=["runs","wickets","strike_rate","economy","catches","batting_avg"]
    labels=["Avg Runs","Avg Wickets","Strike Rate","Economy","Catches","Batting Avg"]

    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=[da[m].mean() for m in metrics],theta=labels,
                                   fill="toself",name=ta,line_color=OR,fillcolor="rgba(255,107,0,0.13)"))
    fig.add_trace(go.Scatterpolar(r=[db[m].mean() for m in metrics],theta=labels,
                                   fill="toself",name=tb,line_color=BL,fillcolor="rgba(79,142,247,0.13)"))
    fig.update_layout(**T,height=460,
                      polar=dict(bgcolor="#060a12",
                                 radialaxis=dict(gridcolor="#1c2333"),
                                 angularaxis=dict(gridcolor="#1c2333")),
                      legend=dict(orientation="h",y=-0.12))
    st.plotly_chart(fig,use_container_width=True)

    c1,c2=st.columns(2)
    for col,team,data,color in [(c1,ta,da,OR),(c2,tb,db,BL)]:
        with col:
            st.markdown(f'<div style="color:{color};font-family:Rajdhani;font-size:1.5rem;font-weight:700;margin-bottom:10px">{team}</div>',unsafe_allow_html=True)
            for m,l in zip(metrics,labels):
                v=round(data[m].mean(),2)
                st.markdown(f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1c2333">'
                             f'<span style="color:#6e7e96">{l}</span><span style="color:{color};font-weight:600">{v}</span></div>',
                             unsafe_allow_html=True)

    st.markdown('<div class="sec">Auction Value Distribution</div>',unsafe_allow_html=True)
    comp=pd.concat([da.assign(Team=ta),db.assign(Team=tb)])
    fig2=px.violin(comp,x="Team",y="auction_value_cr",color="Team",box=True,
                   color_discrete_map={ta:OR,tb:BL})
    fig2.update_layout(**T,height=380,showlegend=False)
    st.plotly_chart(fig2,use_container_width=True)

    st.markdown('<div class="sec">All Teams — Avg Auction Value</div>',unsafe_allow_html=True)
    at=dff.groupby("team")["auction_value_cr"].mean().sort_values(ascending=False).reset_index()
    fig3=px.bar(at,x="team",y="auction_value_cr",color="auction_value_cr",
                color_continuous_scale=["#1c2333",OR,GD],text_auto=".1f")
    fig3.update_traces(textfont_color="white")
    fig3.update_layout(**T,height=360,showlegend=False,coloraxis_showscale=False)
    st.plotly_chart(fig3,use_container_width=True)

# ═══════════════════════════════════════════════════
# LEADERBOARD
# ═══════════════════════════════════════════════════
elif PAGE=="🏆 Leaderboard":
    st.markdown('<div class="hero">LEADERBOARD</div>',unsafe_allow_html=True)
    st.markdown("---")

    cat=st.selectbox("Category",[
        "Most Runs (Career)","Most Wickets (Career)",
        "Best Strike Rate (min 500 runs)","Best Economy Rate (min 50 overs)",
        "Most Fifties","Most Hundreds","Most Catches",
        "Highest Auction Value (Avg)","Most Matches Played", "Highest Score (Inings)"])

    agg=dff.groupby(["player","role","team"]).agg(
        runs=("runs","sum"),wickets=("wickets","sum"),
        sr=("strike_rate","mean"),eco=("economy","mean"),
        val=("auction_value_cr","mean"),matches=("matches","sum"),
        fifties=("fifties","sum"),hundreds=("hundreds","sum"),highest=("highest","max"),
        catches=("catches","sum"),balls=("balls_faced","sum"),
        overs=("overs_bowled","sum")
    ).reset_index()

    CATS={
        "Most Runs (Career)":              ("runs","Runs",False,None,None),
        "Most Wickets (Career)":           ("wickets","Wickets",False,None,None),
        "Best Strike Rate (min 500 runs)": ("sr","Strike Rate",False,"balls",500),
        "Best Economy Rate (min 50 overs)":("eco","Economy",True,"overs",50),
        "Most Fifties":                    ("fifties","Fifties",False,None,None),
        "Most Hundreds":                   ("hundreds","Hundreds",False,None,None),
        "Most Catches":                    ("catches","Catches",False,None,None),
        "Highest Auction Value (Avg)":     ("val","Avg Value (₹Cr)",False,None,None),
        "Most Matches Played":             ("matches","Matches",False,None,None),
        "Highest Score (Inings)":          ("highest","Highest Score",False,None,None),
    }
    col,label,asc,filter_col,filter_min=CATS[cat]
    filt=agg.copy()
    if filter_col and filter_min:
        filt=filt[filt[filter_col]>=filter_min]
    top10=filt.sort_values(col,ascending=asc).head(10).reset_index(drop=True)

    medal={0:"🥇",1:"🥈",2:"🥉"}
    rc={"Batsman":OR,"WK-Batsman":GD,"All-Rounder":GR,"Bowler":BL}

    for i,row in top10.iterrows():
        ri=medal.get(i,f"#{i+1}")
        clr=rc.get(row["role"],OR)
        val_str=f"₹{round(row[col],1)}Cr" if col=="val" else (
            f"{round(row[col],2)}" if col in ["sr","eco"] else str(int(row[col])))
        st.markdown(f"""
        <div class="lbrow">
          <div class="lbrank">{ri}</div>
          <div class="lbname">
            <span style="font-weight:600">{row['player']}</span>
            <span style="color:#6e7e96;font-size:.8rem;margin-left:8px">{row['team']}</span>
            <span class="badge" style="background:{clr}20;color:{clr};border:1px solid {clr}50;margin-left:6px">{row['role']}</span>
          </div>
          <div class="lbval">{val_str}</div>
        </div>""",unsafe_allow_html=True)

    st.markdown('<div class="sec">Chart</div>',unsafe_allow_html=True)
    fig=px.bar(top10.sort_values(col,ascending=not asc),x=col,y="player",
               orientation="h",color="role",text=col,
               color_discrete_map={"Batsman":OR,"WK-Batsman":GD,"All-Rounder":GR,"Bowler":BL})
    fig.update_traces(texttemplate="%{text:.1f}",textfont_color="white")
    fig.update_layout(**T,height=400)
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════
# PREDICTOR
# ═══════════════════════════════════════════════════
elif PAGE=="🤖 Predictor":
    st.markdown('<div class="hero">AUCTION PREDICTOR</div>',unsafe_allow_html=True)
    st.markdown('<div class="sub">Enter stats → Get predicted IPL auction price</div>',unsafe_allow_html=True)
    st.markdown("---")

    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("#### 📋 Player Info")
        role    =st.selectbox("Role",le_r.classes_)
        team    =st.selectbox("Team",le_t.classes_)
        country =st.selectbox("Country",le_c.classes_)
        tier    =st.selectbox("Tier",le_ti.classes_)
        matches =st.slider("Matches",1,16,12)

    with c2:
        st.markdown("#### 🏏 Batting")
        runs       =st.number_input("Runs",0,1200,450)
        balls      =st.number_input("Balls Faced",1,900,320)
        sr         =round((runs/balls)*100,1) if balls else 0.0
        avg        =round(runs/matches,2) if matches else 0.0
        st.metric("Strike Rate",f"{sr}")
        st.metric("Batting Avg",f"{avg}")
        fifties    =st.number_input("Fifties",0,10,2)
        hundreds   =st.number_input("Hundreds",0,5,0)

    with c3:
        st.markdown("#### ⚡ Bowling")
        wickets    =st.number_input("Wickets",0,30,0)
        economy    =st.number_input("Economy",0.0,15.0,8.0,0.1)
        bowl_avg   =st.number_input("Bowling Avg",0.0,80.0,35.0)
        catches    =st.number_input("Catches",0,20,4)

    st.markdown("---")
    if st.button("🔮  PREDICT AUCTION VALUE",use_container_width=True):
        row=np.array([[matches,runs,sr,avg,fifties,hundreds,
                       wickets,economy,bowl_avg,catches,
                       le_r.transform([role])[0],le_t.transform([team])[0],
                       le_c.transform([country])[0],le_ti.transform([tier])[0]]])
        best_p=max(0.2,round(mods[BEST]["model"].predict(row)[0],2))
        all_p={n:max(0.2,round(m["model"].predict(row)[0],2)) for n,m in mods.items()}

        st.markdown(f"""
        <div class="pred">
          <div style="color:#6e7e96;font-size:.85rem;text-transform:uppercase;letter-spacing:2px">Predicted Auction Price</div>
          <div class="pval">₹ {best_p} Cr</div>
          <div style="color:#6e7e96;font-size:.82rem;margin-top:6px">Model: {BEST}</div>
        </div>""",unsafe_allow_html=True)

        st.markdown('<div class="sec">All Models</div>',unsafe_allow_html=True)
        mc=st.columns(3)
        for i,(n,v) in enumerate(all_p.items()):
            kpi(f"₹{v}Cr",n.split(" ",1)[1],mc[i])

        fig=go.Figure(go.Indicator(
            mode="gauge+number+delta",value=best_p,
            delta={"reference":5,"increasing":{"color":GR}},
            title={"text":"Auction Value (₹ Cr)","font":{"color":"#c9d1d9","family":"Rajdhani","size":18}},
            number={"prefix":"₹","suffix":" Cr","font":{"color":GR,"size":48}},
            gauge={"axis":{"range":[0,25],"tickcolor":"#c9d1d9"},
                   "bar":{"color":GR,"thickness":0.35},"bgcolor":"#0d1320",
                   "steps":[{"range":[0,5],"color":"#0d1320"},
                             {"range":[5,12],"color":"#0a180a"},
                             {"range":[12,25],"color":"#102010"}],
                   "threshold":{"line":{"color":OR,"width":4},"value":best_p}}))
        fig.update_layout(**T,height=340)
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════
# ML MODELS
# ═══════════════════════════════════════════════════
elif PAGE=="🔬 ML Models":
    st.markdown('<div class="hero">ML MODEL INSIGHTS</div>',unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sec">Model Performance</div>',unsafe_allow_html=True)
    cols=st.columns(3)
    for i,(n,res) in enumerate(mods.items()):
        c=GR if n==BEST else OR
        with cols[i]:
            st.markdown(f"""
            <div class="kpi" style="border-color:{c}50">
              <div style="font-family:Rajdhani;font-size:1.1rem;font-weight:700;color:{c}">{n}</div>
              <div style="margin:14px 0"><div class="kv" style="color:{c}">{res['r2']}</div>
              <div class="kl">R² Score</div></div>
              <div><span style="color:#6e7e96">MAE: </span><span style="color:{c};font-weight:600">₹{res['mae']} Cr</span></div>
              {"<div style='color:"+GR+";font-size:.75rem;margin-top:8px'>★ BEST MODEL</div>" if n==BEST else ""}
            </div>""",unsafe_allow_html=True)

    st.markdown('<div class="sec">Feature Importance</div>',unsafe_allow_html=True)
    rf=mods["🌲 Random Forest"]["model"]
    fi=pd.DataFrame({"Feature":FEATS,"Importance":rf.feature_importances_}).sort_values("Importance")
    fig=px.bar(fi,x="Importance",y="Feature",orientation="h",
               color="Importance",color_continuous_scale=["#1c2333",BL,OR,GD])
    fig.update_layout(**T,height=440,coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="sec">Predicted vs Actual</div>',unsafe_allow_html=True)
    tmp=df.copy()
    tmp["re"]=LabelEncoder().fit_transform(tmp["role"])
    tmp["te"]=LabelEncoder().fit_transform(tmp["team"])
    tmp["ce"]=LabelEncoder().fit_transform(tmp["country"])
    tmp["tie"]=LabelEncoder().fit_transform(tmp["tier"])
    yhat=rf.predict(tmp[FEATS])
    sc=pd.DataFrame({"Actual":df["auction_value_cr"],"Predicted":yhat,
                     "Player":df["player"],"Role":df["role"]})
    fig2=px.scatter(sc,x="Actual",y="Predicted",color="Role",hover_name="Player",
                    color_discrete_sequence=[OR,GD,GR,BL],opacity=0.7)
    mx=sc[["Actual","Predicted"]].max().max()
    fig2.add_shape(type="line",x0=0,y0=0,x1=mx,y1=mx,line=dict(color=GR,dash="dash",width=1))
    fig2.update_layout(**T,height=420)
    st.plotly_chart(fig2,use_container_width=True)

    st.markdown("""
    <div style="background:#0d1320;border-radius:14px;padding:22px;border:1px solid #1c2333;line-height:2">
    <b style="color:#ffd700">Random Forest</b> — 150 decision trees, averages predictions. Best for this dataset.<br>
    <b style="color:#ffd700">Gradient Boosting</b> — Sequential trees, each fixing previous errors.<br>
    <b style="color:#ffd700">Linear Regression</b> — Baseline model, assumes linear relationships.<br>
    <span style="color:#6e7e96;font-size:.83rem">Features: Matches · Runs · Strike Rate · Batting Avg · Fifties · Hundreds · Wickets · Economy · Bowling Avg · Catches · Role · Team · Country · Tier</span>
    </div>""",unsafe_allow_html=True)
