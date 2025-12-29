import streamlit as st
import pandas as pd
import io

# Set Page Config
st.set_page_config(page_title="Amazon Search Term Optimizer", layout="wide")

st.title("🔍 Amazon Search Term Optimizer: Cannibalization & Harvesting")
st.markdown("""
This tool performs two major analyses on your Search Term Report:
1.  **Cannibalization:** Finds terms appearing in multiple ad groups and suggests where to Keep vs. Negate.
2.  **Harvesting (New!):** Finds high-converting terms (Orders ≥ 2) in **Auto/Broad/Phrase** campaigns that are **NOT** yet targeted as Exact keywords.
""")

# --- HELPER FUNCTIONS ---

def determine_winner(group):
    """
    Decides which row to KEEP based on Sales vs ROAS trade-off.
    """
    max_sales_idx = group['sales_val'].idxmax()
    max_sales_row = group.loc[max_sales_idx]
    
    max_roas_idx = group['calculated_roas'].idxmax()
    max_roas_row = group.loc[max_roas_idx]
    
    if max_sales_idx == max_roas_idx:
        return max_sales_idx, "Best Sales & ROAS"
    
    roas_of_sales_winner = max_sales_row['calculated_roas']
    roas_of_roas_winner = max_roas_row['calculated_roas']
    
    if roas_of_sales_winner == 0:
        return max_roas_idx, "Significantly Better ROAS"

    improvement = (roas_of_roas_winner - roas_of_sales_winner) / roas_of_sales_winner
    
    # 30% Logic
    if improvement > 0.30:
        return max_roas_idx, f"Better ROAS by {improvement:.0%}"
    else:
        return max_sales_idx, "Higher Sales Vol (ROAS diff < 30%)"

def normalize_match_type(val):
    """Normalizes match types to generic categories"""
    if pd.isna(val): return 'UNKNOWN'
    val = str(val).upper()
    if 'EXACT' in val: return 'EXACT'
    if 'PHRASE' in val: return 'PHRASE'
    if 'BROAD' in val: return 'BROAD'
    return 'AUTO/OTHER'

# -------------------------------

# 1. File Upload
uploaded_file = st.file_uploader("Upload Search Term Report (CSV or XLSX)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read File
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        df.columns = df.columns.str.strip()
        
        # 2. Column Mapping
        col_map = {
            'search_term': next((c for c in df.columns if 'Matched product' in c or 'Customer Search Term' in c), None),
            'campaign': next((c for c in df.columns if 'Campaign Name' in c), None),
            'ad_group': next((c for c in df.columns if 'Ad Group Name' in c), None),
            'match_type': next((c for c in df.columns if 'Match Type' in c), None),
            'orders': next((c for c in df.columns if 'Orders' in c or 'Units' in c), None),
            'sales': next((c for c in df.columns if 'Sales' in c), None),
            'spend': next((c for c in df.columns if 'Spend' in c), None),
        }

        if any(v is None for v in col_map.values()):
            st.error(f"Missing columns. Found: {col_map}. Please check your file.")
        else:
            # Data Type Cleanup
            for col in ['orders', 'sales', 'spend']:
                df[col_map[col]] = pd.to_numeric(df[col_map[col]], errors='coerce').fillna(0)
            
            # Normalize Match Type for logic
            df['normalized_match'] = df[col_map['match_type']].apply(normalize_match_type)

            # --- PRE-PROCESSING & AGGREGATION ---
            # We aggregate by Term + Campaign + AdGroup + MatchType
            groupby_cols = [col_map['search_term'], col_map['campaign'], col_map['ad_group'], 'normalized_match']
            
            df_agg = df.groupby(groupby_cols, as_index=False).agg({
                col_map['orders']: 'sum',
                col_map['sales']: 'sum',
                col_map['spend']: 'sum'
            })
            
            df_agg.rename(columns={col_map['sales']: 'sales_val', col_map['spend']: 'spend_val'}, inplace=True)
            df_agg['calculated_roas'] = df_agg.apply(lambda x: x['sales_val']/x['spend_val'] if x['spend_val'] > 0 else 0, axis=1)

            # ==========================================
            # ANALYSIS 1: CANNIBALIZATION (Existing Logic)
            # ==========================================
            st.header("1. ⚔️ Keyword Cannibalization")
            st.info("Search terms appearing in multiple ad groups. 'Negate' the inefficient ones.")

            sales_df = df_agg[df_agg[col_map['orders']] > 0].copy()
            cannibal_counts = sales_df.groupby(col_map['search_term']).size()
            cannibal_terms = cannibal_counts[cannibal_counts > 1].index.tolist()

            if not cannibal_terms:
                st.success("No cannibalization found.")
            else:
                c_results = []
                for term in cannibal_terms:
                    term_data = sales_df[sales_df[col_map['search_term']] == term].copy()
                    winner_idx, reason = determine_winner(term_data)
                    
                    for idx, row in term_data.iterrows():
                        is_winner = (idx == winner_idx)
                        rec = f"✅ KEEP ({reason})" if is_winner else "❌ NEGATE"
                        c_results.append({
                            "Search Term": term,
                            "Campaign": row[col_map['campaign']],
                            "Ad Group": row[col_map['ad_group']],
                            "Type": row['normalized_match'],
                            "Spend": row['spend_val'],
                            "Orders": row[col_map['orders']],
                            "ROAS": round(row['calculated_roas'], 2),
                            "Action": rec
                        })
                
                c_df = pd.DataFrame(c_results).sort_values(by=['Search Term', 'Orders'], ascending=[True, False])
                st.dataframe(c_df.style.apply(lambda x: ['background-color: #ffd2d2' if 'NEGATE' in v else '' for v in x], subset=['Action']), use_container_width=True)
                
                # Download Cannibalization
                st.download_button("Download Cannibalization Report", c_df.to_csv(index=False).encode('utf-8'), "cannibalization.csv")


            st.markdown("---")


            # ==========================================
            # ANALYSIS 2: HARVESTING (New Logic)
            # ==========================================
            st.header("2. 🚀 Growth Opportunities (Harvesting)")
            st.info("High-performing search terms (Orders ≥ 2) from Auto/Broad/Phrase that are **NOT** currently targeted as Exact.")

            # 1. Identify Existing Exact Terms
            # We look at the WHOLE report. If a term appears as 'EXACT' anywhere, we assume it's covered.
            existing_exact_terms = set(df_agg[df_agg['normalized_match'] == 'EXACT'][col_map['search_term']].str.lower().unique())

            # 2. Identify Candidates from Broad/Phrase/Auto
            candidates = df_agg[
                (df_agg['normalized_match'].isin(['BROAD', 'PHRASE', 'AUTO/OTHER'])) & 
                (df_agg[col_map['orders']] >= 2)
            ].copy()

            # 3. Filter: Remove if it already exists in Exact list
            harvest_opportunities = candidates[~candidates[col_map['search_term']].str.lower().isin(existing_exact_terms)]

            if harvest_opportunities.empty:
                st.write("No new harvesting opportunities found (all high-sales terms are already Exact targets).")
            else:
                h_results = []
                for idx, row in harvest_opportunities.iterrows():
                    h_results.append({
                        "Search Term": row[col_map['search_term']],
                        "Found In (Campaign)": row[col_map['campaign']],
                        "Found In (Ad Group)": row[col_map['ad_group']],
                        "Match Type": row['normalized_match'],
                        "Orders": row[col_map['orders']],
                        "Sales": row['sales_val'],
                        "ROAS": round(row['calculated_roas'], 2),
                        "Recommendation 1": "🎯 ADD to Manual Exact",
                        "Recommendation 2": "⛔ NEGATE from Source"
                    })
                
                h_df = pd.DataFrame(h_results).sort_values(by='Sales', ascending=False)
                
                # Metrics
                col1, col2 = st.columns(2)
                col1.metric("New Keywords to Harvest", len(h_df))
                col2.metric("Potential Sales Revenue", f"₹ {h_df['Sales'].sum():,.2f}")

                st.dataframe(h_df, use_container_width=True)
                
                # Download Harvesting
                st.download_button("Download Harvesting Plan", h_df.to_csv(index=False).encode('utf-8'), "harvesting_plan.csv")
                
                # Actionable List for Copy-Paste
                with st.expander("View Quick Copy-Paste Lists"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Terms to ADD (Exact)")
                        st.text_area("Copy these", "\n".join(h_df['Search Term'].unique()), height=300)
                    with c2:
                        st.subheader("Pairs to NEGATE (From Source)")
                        st.write("Go to these Ad Groups and negate the specific term:")
                        st.dataframe(h_df[['Found In (Campaign)', 'Found In (Ad Group)', 'Search Term']], hide_index=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
