"""
preprocess.py — Converts raw Kaggle IPL CSVs into ipl_data.csv for the app.
Run ONCE before starting the app:  python3 preprocess.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("📂 Loading Kaggle datasets...")
matches    = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

# ── Clean season names ────────────────────────────────────────────────────────
season_map = {
    "2007/08": 2008, "2009/10": 2010, "2020/21": 2021,
}
matches["season"] = matches["season"].replace(season_map)
matches["season"] = pd.to_numeric(matches["season"], errors="coerce").fillna(0).astype(int)
deliveries = deliveries.merge(matches[["id","season"]], left_on="match_id", right_on="id", how="left")

# ── Team name normalisation ───────────────────────────────────────────────────
TEAM_MAP = {
    "Delhi Daredevils":          "DC",
    "Delhi Capitals":            "DC",
    "Kings XI Punjab":           "PBKS",
    "Punjab Kings":              "PBKS",
    "Royal Challengers Bangalore":"RCB",
    "Royal Challengers Bengaluru":"RCB",
    "Rising Pune Supergiant":    "RPS",
    "Rising Pune Supergiants":   "RPS",
    "Mumbai Indians":            "MI",
    "Chennai Super Kings":       "CSK",
    "Kolkata Knight Riders":     "KKR",
    "Rajasthan Royals":          "RR",
    "Sunrisers Hyderabad":       "SRH",
    "Deccan Chargers":           "DC_old",
    "Gujarat Titans":            "GT",
    "Gujarat Lions":             "GL",
    "Lucknow Super Giants":      "LSG",
    "Kochi Tuskers Kerala":      "KTK",
    "Pune Warriors":             "PWI",
}
for col in ["batting_team","bowling_team"]:
    deliveries[col] = deliveries[col].replace(TEAM_MAP)
for col in ["team1","team2","winner","toss_winner"]:
    matches[col] = matches[col].replace(TEAM_MAP)

# ── Batting stats per player per season ───────────────────────────────────────
print("🏏 Computing batting stats...")
bat = deliveries.groupby(["batter","batting_team","season"]).agg(
    runs        = ("batsman_runs","sum"),
    balls_faced = ("batsman_runs","count"),
    matches_bat = ("match_id","nunique"),
    fours       = ("batsman_runs", lambda x: (x==4).sum()),
    sixes       = ("batsman_runs", lambda x: (x==6).sum()),
).reset_index().rename(columns={"batter":"player","batting_team":"team"})

bat["strike_rate"] = (bat["runs"] / bat["balls_faced"] * 100).round(2)

# Fifties and hundreds — need innings-level data
inns = deliveries.groupby(["batter","match_id","season"])["batsman_runs"].sum().reset_index()
inns.columns = ["player","match_id","season","inns_runs"]
milestones = inns.groupby(["player","season"]).apply(
    lambda x: pd.Series({
        "fifties":  ((x["inns_runs"] >= 50) & (x["inns_runs"] < 100)).sum(),
        "hundreds": (x["inns_runs"] >= 100).sum(),
        "highest":  x["inns_runs"].max(),
    })
).reset_index()

bat = bat.merge(milestones, on=["player","season"], how="left")
bat["fifties"]  = bat["fifties"].fillna(0).astype(int)
bat["hundreds"] = bat["hundreds"].fillna(0).astype(int)
bat["highest"]  = bat["highest"].fillna(0).astype(int)
bat["batting_avg"] = (bat["runs"] / bat["matches_bat"]).round(2)

# ── Bowling stats per player per season ───────────────────────────────────────
print("⚡ Computing bowling stats...")
bowl_df = deliveries[deliveries["is_wicket"] == 1]
# Exclude run-outs from bowler credit
bowl_df = bowl_df[~bowl_df["dismissal_kind"].isin(["run out","retired hurt","obstructing the field"])]

wickets = bowl_df.groupby(["bowler","bowling_team","season"])["is_wicket"].sum().reset_index()
wickets.columns = ["player","team","season","wickets"]

economy_data = deliveries.groupby(["bowler","bowling_team","season"]).agg(
    runs_given  = ("total_runs","sum"),
    balls_bowled= ("total_runs","count"),
    matches_bowl= ("match_id","nunique"),
).reset_index().rename(columns={"bowler":"player","bowling_team":"team"})

economy_data["overs_bowled"] = (economy_data["balls_bowled"] / 6).round(1)
economy_data["economy"]      = (economy_data["runs_given"] / economy_data["overs_bowled"]).round(2)
economy_data["economy"]      = economy_data["economy"].replace([np.inf, -np.inf], 0)

bowl = economy_data.merge(wickets, on=["player","team","season"], how="left")
bowl["wickets"] = bowl["wickets"].fillna(0).astype(int)
bowl["bowling_avg"] = np.where(
    bowl["wickets"] > 0,
    (bowl["runs_given"] / bowl["wickets"]).round(2),
    99.0
)

# ── Fielding ──────────────────────────────────────────────────────────────────
catches_df = deliveries[deliveries["dismissal_kind"] == "caught"]
catches = catches_df.groupby(["fielder","season"])["is_wicket"].sum().reset_index()
catches.columns = ["player","season","catches"]

# ── Merge batting + bowling ───────────────────────────────────────────────────
print("🔀 Merging stats...")
df = bat.merge(
    bowl[["player","season","wickets","overs_bowled","economy","bowling_avg","matches_bowl"]],
    on=["player","season"], how="outer"
)
df = df.merge(catches, on=["player","season"], how="left")

# Fill NAs
df["runs"]         = df["runs"].fillna(0).astype(int)
df["balls_faced"]  = df["balls_faced"].fillna(0).astype(int)
df["strike_rate"]  = df["strike_rate"].fillna(0.0)
df["batting_avg"]  = df["batting_avg"].fillna(0.0)
df["fifties"]      = df["fifties"].fillna(0).astype(int)
df["hundreds"]     = df["hundreds"].fillna(0).astype(int)
df["highest"]      = df["highest"].fillna(0).astype(int)
df["wickets"]      = df["wickets"].fillna(0).astype(int)
df["overs_bowled"] = df["overs_bowled"].fillna(0.0)
df["economy"]      = df["economy"].fillna(0.0)
df["bowling_avg"]  = df["bowling_avg"].fillna(99.0)
df["catches"]      = df["catches"].fillna(0).astype(int)
df["matches_bat"]  = df["matches_bat"].fillna(0).astype(int)
df["matches_bowl"] = df["matches_bowl"].fillna(0).astype(int)
df["matches"]      = df[["matches_bat","matches_bowl"]].max(axis=1).astype(int)

# Team: fill from bowling side if missing from batting
df["team"] = df["team_x"].combine_first(df["team_y"]) if "team_y" in df.columns else df["team"]
if "team_x" in df.columns: df.drop(columns=["team_x","team_y"], inplace=True, errors="ignore")

# ── Role assignment ───────────────────────────────────────────────────────────
def assign_role(row):
    WK = ["MS Dhoni","KL Rahul","Rishabh Pant","Sanju Samson","Jos Buttler",
          "Ishan Kishan","KD Karthik","Wriddhiman Saha","AB de Villiers",
          "Quinton de Kock","Nicholas Pooran","Heinrich Klaasen","RV Uthappa",
          "PA Patel","NV Ojha","CMK Nair","SS Iyer","DA Miller"]
    if row["player"] in WK:
        return "WK-Batsman"
    r = row["runs"]; w = row["wickets"]
    if r >= 200 and w >= 8:   return "All-Rounder"
    if r >= 100 and w >= 15:  return "All-Rounder"
    if w >= 10 and r < 100:   return "Bowler"
    if r >= 150:              return "Batsman"
    if w >= 8:                return "Bowler"
    return "All-Rounder"

df["role"] = df.apply(assign_role, axis=1)

# ── Country mapping ───────────────────────────────────────────────────────────
COUNTRY = {
    "V Kohli":"India","RG Sharma":"India","MS Dhoni":"India","SK Raina":"India",
    "S Dhawan":"India","KL Rahul":"India","Rishabh Pant":"India","Hardik Pandya":"India",
    "YS Chahal":"India","JJ Bumrah":"India","RA Jadeja":"India","R Ashwin":"India",
    "KD Karthik":"India","RV Uthappa":"India","Ishan Kishan":"India","Shubman Gill":"India",
    "DA Warner":"Australia","GJ Maxwell":"Australia","SR Watson":"Australia",
    "CLR Warner":"Australia","AC Gilchrist":"Australia","SK Warne":"Australia",
    "AB de Villiers":"SA","F du Plessis":"SA","Kagiso Rabada":"SA","HH Gibbs":"SA",
    "CH Gayle":"WI","DJ Bravo":"WI","SP Narine":"WI","KA Pollard":"WI","DR Smith":"WI",
    "BB McCullum":"NZ","MJ Guptill":"NZ","KS Williamson":"NZ","TG Southee":"NZ",
    "Mohammad Nabi":"Afghanistan","Rashid Khan":"Afghanistan","Mujeeb Ur Rahman":"Afghanistan",
    "JC Buttler":"England","BA Stokes":"England","EJG Morgan":"England","JM Bairstow":"England",
    "SL Malinga":"Sri Lanka","NLTC Perera":"Sri Lanka","AD Russell":"WI",
    "Shakib Al Hasan":"Bangladesh","Mustafizur Rahman":"Bangladesh",
}
df["country"] = df["player"].map(COUNTRY).fillna("India")

# ── Tier ─────────────────────────────────────────────────────────────────────
TIER_S = ["V Kohli","RG Sharma","MS Dhoni","AB de Villiers","DA Warner","CH Gayle",
          "JJ Bumrah","SP Narine","Rashid Khan","KL Rahul","Rishabh Pant",
          "Hardik Pandya","YS Chahal","RA Jadeja","DJ Bravo","R Ashwin",
          "GJ Maxwell","KA Pollard","KS Williamson","F du Plessis","Kagiso Rabada"]
TIER_A = ["SK Raina","S Dhawan","KD Karthik","RV Uthappa","Shubman Gill",
          "Ishan Kishan","JC Buttler","BA Stokes","B Kumar","SL Malinga",
          "PP Chawla","A Mishra","JM Bairstow","AD Russell","EJG Morgan"]
df["tier"] = df["player"].apply(
    lambda p: "S" if p in TIER_S else ("A" if p in TIER_A else "B")
)

# ── Auction value formula ─────────────────────────────────────────────────────
TIER_M = {"S":1.8,"A":1.2,"B":0.75}
def auction_val(row):
    tm = TIER_M[row["tier"]]
    base = 1.5
    role = row["role"]
    r,sr,w,eco,avg,f,h,c = (row["runs"],row["strike_rate"],row["wickets"],
                              row["economy"],row["batting_avg"],row["fifties"],
                              row["hundreds"],row["catches"])
    if role in ["Batsman","WK-Batsman"]:
        v = base + (r/90) + (sr/45) + f*0.4 + h*1.8 + c*0.1
    elif role == "Bowler":
        eco_s = max(eco, 5)
        v = base + w*0.35 + (10/eco_s) + c*0.1
    else:
        eco_s = max(eco, 5)
        v = base + (r/130) + w*0.28 + (sr/90) + (8/eco_s)
    import random
    v = v * tm + random.uniform(-0.3, 0.3)
    return round(max(0.2, min(25.0, v)), 2)

import random; random.seed(42)
df["auction_value_cr"] = df.apply(auction_val, axis=1)

# ── Final cleanup ─────────────────────────────────────────────────────────────
df = df[df["season"] > 2000]
df = df[df["player"].notna() & (df["player"] != "")]
df = df[df["matches"] >= 3]

final_cols = ["player","role","team","country","tier","season","matches",
              "runs","balls_faced","strike_rate","batting_avg","fifties","hundreds","highest",
              "wickets","overs_bowled","economy","bowling_avg","catches","auction_value_cr"]
df = df[final_cols].reset_index(drop=True)

df.to_csv("ipl_data.csv", index=False)
print(f"\n✅ ipl_data.csv created!")
print(f"   Records : {len(df)}")
print(f"   Players : {df['player'].nunique()}")
print(f"   Seasons : {sorted(df['season'].unique())}")
print(f"\nTop 5 run scorers:")
print(df.groupby("player")["runs"].sum().nlargest(5))
print(f"\nTop 5 wicket takers:")
print(df.groupby("player")["wickets"].sum().nlargest(5))
