import streamlit as st
import pandas as pd
import io

# Set Page Config
st.set_page_config(page_title="Amazon Search Term Cannibalization Tool", layout="wide")

st.title("🔍 Amazon Search Term Cannibalization & Negation Analyzer")
st.markdown("""
Upload your **Sponsored Products Search Term Report** (CSV or Excel).
**Logic Used:**
1. Identifies search terms appearing in multiple ad groups.
2. Compares **Sales Volume** vs. **ROAS**.
3. **Winner Rule:** It keeps the Highest Sales target UNLESS the alternative has a **ROAS > 30% better**, in which case it prioritizes Efficiency.
""")

# --- DECISION LOGIC FUNCTION ---
def determine_winner(group):
    """
    Decides which row to KEEP based on Sales vs ROAS trade-off.
    Returns the index of the winner.
    """
    # 1. Find the row with Max Sales
    max_sales_idx = group['sales_val'].idxmax()
    max_sales_row = group.loc[max_sales_idx]
    
    # 2. Find the row with Max ROAS
    max_roas_idx = group['calculated_roas'].idxmax()
    max_roas_row = group.loc[max_roas_idx]
    
    # 3. Compare
    # If the same row has both max sales and max roas, it's the winner.
    if max_sales_idx == max_roas_idx:
        return max_sales_idx, "Best Sales & ROAS"
    
    # If different, apply the 30% threshold rule
    # Get the ROAS of the 'Sales Winner' to compare against the 'ROAS Winner'
    roas_of_sales_winner = max_sales_row['calculated_roas']
    roas_of_roas_winner = max_roas_row['calculated_roas']
    
    # Avoid division by zero issues
    if roas_of_sales_winner == 0:
        return max_roas_idx, "Significantly Better ROAS"

    # Calculate percentage improvement
    improvement = (roas_of_roas_winner - roas_of_sales_winner) / roas_of_sales_winner
    
    # If ROAS winner is > 30% better than the Sales winner, choose ROAS (Efficiency)
    if improvement > 0.30:
        return max_roas_idx, f"Better ROAS by {improvement:.0%}"
    else:
        # Otherwise, stick with the higher volume (Sales)
        return max_sales_idx, "Higher Sales Vol (ROAS diff < 30%)"

# -------------------------------

# 1. File Upload
uploaded_file = st.file_uploader("Upload Search Term Report", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            # Requires 'openpyxl' library
            df = pd.read_excel(uploaded_file)
            
        # Clean Column Names
        df.columns = df.columns.str.strip()
        
        # Identify key columns dynamically
        col_map = {
            'search_term': next((c for c in df.columns if 'Matched product' in c or 'Customer Search Term' in c), None),
            'campaign': next((c for c in df.columns if 'Campaign Name' in c), None),
            'ad_group': next((c for c in df.columns if 'Ad Group Name' in c), None),
            'orders': next((c for c in df.columns if 'Orders' in c or 'Units' in c), None),
            'sales': next((c for c in df.columns if 'Sales' in c), None),
            'spend': next((c for c in df.columns if 'Spend' in c), None),
        }

        # Check for missing columns
        missing = [k for k, v in col_map.items() if v is None]
        if missing:
            st.error(f"Missing required columns: {missing}. Please check your file format.")
        else:
            # Data Preparation
            # Standardize numeric columns
            for col in ['orders', 'sales', 'spend']:
                df[col_map[col]] = pd.to_numeric(df[col_map[col]], errors='coerce').fillna(0)

            # 2. AGGREGATION STEP
            # Group by Search Term, Campaign, and Ad Group to combine daily rows
            groupby_cols = [col_map['search_term'], col_map['campaign'], col_map['ad_group']]
            
            df_aggregated = df.groupby(groupby_cols, as_index=False).agg({
                col_map['orders']: 'sum',
                col_map['sales']: 'sum',
                col_map['spend']: 'sum'
            })
            
            # Rename for easier access in the logic function
            df_aggregated.rename(columns={
                col_map['sales']: 'sales_val',
                col_map['spend']: 'spend_val'
            }, inplace=True)

            # Calculate ROAS
            df_aggregated['calculated_roas'] = df_aggregated.apply(
                lambda x: x['sales_val'] / x['spend_val'] if x['spend_val'] > 0 else 0, axis=1
            )

            # Filter for Search Terms with at least one sale
            sales_df = df_aggregated[df_aggregated[col_map['orders']] > 0].copy()
            
            # 3. Find Cannibalization
            cannibal_counts = sales_df.groupby(col_map['search_term']).size()
            cannibal_terms = cannibal_counts[cannibal_counts > 1].index.tolist()
            
            if not cannibal_terms:
                st.success("No keyword cannibalization found! All converting search terms are unique.")
            else:
                st.warning(f"Found {len(cannibal_terms)} search terms appearing in multiple campaigns/ad groups.")
                
                results = []
                
                # Analyze each term
                for term in cannibal_terms:
                    # Get all rows for this specific search term
                    term_data = sales_df[sales_df[col_map['search_term']] == term].copy()
                    
                    # APPLY DECISION LOGIC
                    winner_idx, reason = determine_winner(term_data)
                    
                    for idx, row in term_data.iterrows():
                        is_winner = (idx == winner_idx)
                        recommendation = f"✅ KEEP ({reason})" if is_winner else "❌ NEGATE"
                        
                        results.append({
                            "Search Term": term,
                            "Campaign Name": row[col_map['campaign']],
                            "Ad Group Name": row[col_map['ad_group']],
                            "Orders": row[col_map['orders']],
                            "Sales": row['sales_val'],
                            "Spend": row['spend_val'],
                            "ROAS": round(row['calculated_roas'], 2),
                            "Recommendation": recommendation
                        })
                
                # Create Final Report
                final_report = pd.DataFrame(results)
                
                # Sort for readability: Put Negates next to their Keeps
                final_report = final_report.sort_values(by=['Search Term', 'Sales'], ascending=[True, False])

                # Metrics
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Converting Search Terms", len(sales_df[col_map['search_term']].unique()))
                c2.metric("Cannibalized Terms", len(cannibal_terms))
                negate_spend = final_report[final_report['Recommendation'].str.contains('NEGATE')]['Spend'].sum()
                c3.metric("Redundant Spend (Negate)", f"₹ {negate_spend:,.2f}")

                # Display Main Table
                st.subheader("Cannibalization Analysis (Sales vs ROAS)")
                st.dataframe(
                    final_report.style.apply(
                        lambda x: ['background-color: #d4edda' if 'KEEP' in v else 'background-color: #f8d7da' for v in x], 
                        subset=['Recommendation']
                    ), 
                    use_container_width=True
                )
                
                # Download CSV
                csv = final_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Recommendations (CSV)",
                    data=csv,
                    file_name="cannibalization_recommendations.csv",
                    mime="text/csv",
                )

                # Negation List
                st.divider()
                st.subheader("Action Plan: Negation List")
                negate_only = final_report[final_report['Recommendation'].str.contains('NEGATE')]
                st.write("Add these as **Negative Exact** in the specific campaigns below:")
                st.table(negate_only[['Search Term', 'Campaign Name', 'Ad Group Name', 'Spend', 'ROAS']])

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a CSV or Excel file to begin.")
