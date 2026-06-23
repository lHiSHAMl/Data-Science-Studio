#reporting.py
import streamlit as st
import pandas as pd
import io
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import reporting_lib.reports_execution as reports_execution
# NOTE: You will need to install fpdf2 for the download button to work later
# from fpdf import FPDF 

def run():
    # 1. Initialize session state for report and cache
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = {}
    st.session_state.pipeline.setdefault("report_items", [])

    # The cache is NEVER saved, only exists during the session
    if 'report_cache' not in st.session_state:
        st.session_state.report_cache = {}

    st.title("📄 Comprehensive Project Report")
    
    report_items = st.session_state.pipeline["report_items"]

    if not report_items:
        st.info("No report items have been added yet. Please configure items from the Machine Learning, Statistics, or Visualization pages.")
        return

    # --- Header and Download (PDF Placeholder) ---
    st.markdown("### Report Content Preview")
    
    # Placeholder for PDF Download Button
    st.warning("Download functionality requires the 'fpdf2' library and is currently a placeholder.")
    # if st.button("Download Final Report (PDF)"):
    #     # ... PDF generation logic using the cache ...
    #     pass

    # --- Iteration and Execution ---
    for item_block in report_items:
    
        source_type = item_block['type']
        
        # Determine the source data to pass to the execution function
        # For ML, we use the metrics_snapshot from the indexed ML step
        source_name = item_block['pipeline_step_name']

        source_data = {}
        
        if source_type == "Machine Learning":
            # Access the correct model's snapshot from the ML list
            try:
                source_data_index = item_block['pipeline_source_index']

                source_data = st.session_state.pipeline['ML'][source_data_index]['metrics_snapshot']
            except (IndexError, KeyError):
                st.error(f"Could not find source data for {source_name}. It might have been deleted.")
                continue
        if source_type == "DF":
            try:
                source_data = item_block['df']
            except KeyError:
                st.error(f"Could not find source data for {source_name}. It might have been deleted.")
                continue
        elif source_type == "Visualization":
            # Handle visualization items
            try:
                source_data_index = item_block['pipeline_source_index']

                # Get visualization data from session state
                viz_item = st.session_state.visualizations[source_data_index]
                source_data = {
                    'viz_config': viz_item['config'],
                    'viz_comments': viz_item.get('comments', ''),
                    'viz_id': viz_item['id']
                }
            except (IndexError, KeyError):
                st.error(f"Could not find visualization data for {source_name}. It might have been deleted.")
                continue

        # --- Report Item Block: Expander ---
        # st.success(st.session_state.pipeline['ML'][source_data_index].get('comments',  ""))
        with st.expander(f"✨ {source_type}: {source_name}", expanded=True):
            if source_type == "DF":
                        st.markdown(f"**Comment:** {item_block.get('comment', 'No comment provided.')}")    
                        st.dataframe(source_data)
            # elif  source_type == "Machine Learning":
            else :
                # Sub-items within the block (ordered by user)
                    for i, choice in enumerate(item_block['choices']):
                        
                        function_name = choice['function_name']
                        user_title = choice['user_title']
                        
                        # Caching Key
                        cache_key = f"{item_block['id']}_{function_name}"
                        
                        # --- Execution and Caching ---
                        if cache_key in st.session_state.report_cache:
                            asset = st.session_state.report_cache[cache_key]
                        else:
                            # Run the expensive generation function
                            try:
                                asset = reports_execution.execute_report_asset(
                                    item_type=source_name,
                                    function_name=function_name,
                                    source_data=source_data
                                )
                                # Store in cache
                                st.session_state.report_cache[cache_key] = asset
                            except Exception as e:
                                st.error(f"Error generating asset '{user_title}': {e}")
                                continue

                        # --- Rendering the Asset ---
                        st.markdown(f"#### 🏷️ {user_title}")
                        
                        asset_type = asset['type']
                        
                        if asset_type == 'text':
                            st.markdown(asset['content'], unsafe_allow_html=True)
                        elif asset_type == 'plot':
                            # Ensure the plot figure is closed after rendering
                            st.pyplot(asset['content'])
                            plt.close(asset['content']) 
                        elif asset_type == 'dataframe':
                            st.dataframe(asset['content'])
                        elif asset_type == 'plotly':
                            st.plotly_chart(
                                asset['content'],
                                use_container_width=True,
                                key=f"plotly_{item_block['id']}_{function_name}_{i}"
                            )

                        else:
                            st.warning(f"Unknown asset type: {asset_type}")


        # --- Delete Item Button ---
        if st.button(f"🗑️ Remove {source_name} from Report", key=f"del_{item_block['id']}"):
            # Find and remove the item
            st.session_state.pipeline["report_items"] = [
                item for item in st.session_state.pipeline["report_items"] 
                if item['id'] != item_block['id']
            ]
            # Invalidate associated cache entries (optional, but clean)
            if item_block['type'] != "DF":
                keys_to_delete = [f"{item_block['id']}_{c['function_name']}" for c in item_block['choices']]
                for k in keys_to_delete:
                    if k in st.session_state.report_cache:
                        del st.session_state.report_cache[k]
                    
            st.rerun()

    # --- Final Order Adjustment (Optional, Advanced) ---
    st.markdown("---")
    st.markdown("#### Adjust Report Block Order")
    
    # Display simplified list for reordering blocks
    new_order_list = []
    
    for i, item in enumerate(report_items):
        col1, col2 = st.columns([10, 1])
        
        with col1:
            st.markdown(f"**{i + 1}.** {item['pipeline_step_name']} ({item['type']})")
        
        with col2:
            if i > 0 and st.button("Move Up", key=f"main_up_{i}"):
                report_items[i], report_items[i-1] = report_items[i-1], report_items[i]
                st.rerun()
            
            if i < len(report_items) - 1 and st.button("Move Down", key=f"main_down_{i}"):
                report_items[i], report_items[i+1] = report_items[i+1], report_items[i]
                st.rerun()

# We include the original model_script.py content as requested for completeness.
