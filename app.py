import streamlit as st
import pandas as pd
import io

# Set Page Config
st.set_page_config(page_title="Amazon Search Term Cannibalization Tool", layout="wide")

st.title("🔍 Amazon Search Term Cannibalization & Negation Analyzer")
st.markdown("""
Upload your **Sponsored Products Search Term Report** (CSV or Excel) to identify search terms that are appearing in multiple campaigns/ad groups. 
The app will recommend which ones to **Keep** and which to **Negate** based on performance.
""")

# 1. File Upload - Updated to accept CSV and XLSX
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
            'orders': next((c for c in df.columns if 'Orders' in c or 'Units' in c), None), # Broad matching for orders
            'sales': next((c for c in df.columns if 'Sales' in c), None),
            'spend': next((c for c in df.columns if 'Spend' in c), None),
        }

        # Check for missing columns
        missing = [k for k, v in col_map.items() if v is None]
        if missing:
            st.error(f"Missing required columns: {missing}. Please check your file format.")
        else:
            # Data Preparation
            # Convert numeric columns to numbers, forcing errors to NaN then 0
            for col in ['orders', 'sales', 'spend']:
                df[col_map[col]] = pd.to_numeric(df[col_map[col]], errors='coerce').fillna(0)

            # 2. AGGREGATION STEP (Crucial for Daily Reports)
            # Group by Search Term, Campaign, and Ad Group to combine daily rows into one total per ad group
            groupby_cols = [col_map['search_term'], col_map['campaign'], col_map['ad_group']]
            
            df_aggregated = df.groupby(groupby_cols, as_index=False).agg({
                col_map['orders']: 'sum',
                col_map['sales']: 'sum',
                col_map['spend']: 'sum'
            })

            # Filter for Search Terms with at least one sale
            sales_df = df_aggregated[df_aggregated[col_map['orders']] > 0].copy()
            
            # 3. Find Cannibalization
            # Check if the search term appears in more than 1 Campaign/Ad Group combination
            cannibal_counts = sales_df.groupby(col_map['search_term']).size()
            cannibal_terms = cannibal_counts[cannibal_counts > 1].index.tolist()
            
            if not cannibal_terms:
                st.success("No keyword cannibalization found! All converting search terms are unique to their respective ad groups.")
            else:
                st.warning(f"Found {len(cannibal_terms)} search terms appearing in multiple campaigns/ad groups.")
                
                results = []
                
                for term in cannibal_terms:
                    term_data = sales_df[sales_df[col_map['search_term']] == term].copy()
                    
                    # Calculate ROAS
                    # Avoid division by zero
                    term_data['calculated_roas'] = term_data.apply(
                        lambda x: x[col_map['sales']] / x[col_map['spend']] if x[col_map['spend']] > 0 else 0, axis=1
                    )
                    
                    # Sort by ROAS descending to find the "Winner"
                    term_data = term_data.sort_values(by='calculated_roas', ascending=False)
                    
                    # Logic: Top ROAS is Keep, others are Negate
                    for i, (index, row) in enumerate(term_data.iterrows()):
                        recommendation = "✅ KEEP (Best Performance)" if i == 0 else "❌ NEGATE (Cannibalization)"
                        
                        results.append({
                            "Search Term": term,
                            "Campaign Name": row[col_map['campaign']],
                            "Ad Group Name": row[col_map['ad_group']],
                            "Orders": row[col_map['orders']],
                            "Sales": row[col_map['sales']],
                            "Spend": row[col_map['spend']],
                            "ROAS": round(row['calculated_roas'], 2),
                            "Recommendation": recommendation
                        })
                
                # Create Report DataFrame
                final_report = pd.DataFrame(results)
                
                # Metrics
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Converting Search Terms", len(sales_df[col_map['search_term']].unique()))
                c2.metric("Cannibalized Terms", len(cannibal_terms))
                negate_spend = final_report[final_report['Recommendation'].str.contains('NEGATE')]['Spend'].sum()
                c3.metric("Redundant Spend", f"₹ {negate_spend:,.2f}")

                # Display Main Table
                st.subheader("Cannibalization Analysis")
                st.dataframe(final_report, use_container_width=True)
                
                # Download CSV
                csv = final_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Recommendations (CSV)",
                    data=csv,
                    file_name="cannibalization_recommendations.csv",
                    mime="text/csv",
                )

                # Negation List for Copy-Paste
                st.divider()
                st.subheader("Action Plan: Negation List")
                st.info("Add these search terms as **Negative Exact** in the specific Campaign > Ad Group listed below.")
                
                negate_only = final_report[final_report['Recommendation'].str.contains('NEGATE')]
                st.table(negate_only[['Search Term', 'Campaign Name', 'Ad Group Name', 'Spend']])

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("Please upload a file to begin.")
