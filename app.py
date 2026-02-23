# Complete fixed app.py

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ElectricalLoad:
    """Base class for electrical loads"""
    name: str
    load_kva: float
    diversity_factor: float = 1.0
    
    @property
    def demand_load(self) -> float:
        return self.load_kva * self.diversity_factor

@dataclass
class ResidentialUnit:
    unit_type: str
    load_kva: float
    quantity: int = 0
    
    @property
    def total_load(self) -> float:
        return self.load_kva * self.quantity

@dataclass
class RetailShop:
    name: str
    area_sqm: float
    breaker_size: str
    load_kw: float
    power_factor: float = 0.85
    
    @property
    def load_kva(self) -> float:
        return self.load_kw / self.power_factor
    
    def __post_init__(self):
        """Validate the data after initialization"""
        if self.area_sqm < 0:
            raise ValueError("Area cannot be negative")
        if self.load_kw < 0:
            raise ValueError("Load cannot be negative")

# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class ElectricalLoadCalculator:
    """Main calculator class for electrical loads"""
    
    def __init__(self):
        self.residential_units = {
            "Studio Apartment (1-room)": 3.5,
            "Studio Apartment (2-room)": 3.5,
            "2-room flat": 3.5,
            "3-room flat": 3.8,
            "4-room flat": 4.3,
            "5-room flat": 4.5
        }
        
        self.common_installations = {
            "Escalator": 22,
            "Service Lift (20 man)": 35,
            "Lift 11 sty (13 man)": 20,
            "Lift (20 man)": 35,
            "Public Lighting Circuit": 12,
            "Outdoor Lighting Circuit": 17,
            "Domestic Booster Pump": 3.52,
            "Fire Hose Reel Pump": 2.6,
            "Refuse Handling Plant": 10,
            "Wet Riser System": 25,
            "Sprinkler System": 44,
            "Mech Ventilation": 11
        }
        
        self.facilities = {
            "Future Communal Facilities": 42,
            "Bin Centre": 10,
            "Shop (per 100sqm)": 42,
            "Hawker Centre": 676.88,
            "RC Centre": 42,
            "Polyclinic": 881.28,
            "MNO": 41.6,
            "EPS": 14,
            "KDC": 104,
            "CC": 693,
            "Mech": 762
        }
        
    def calculate_residential_load(self, unit_counts: Dict[str, int]) -> float:
        """Calculate total residential load"""
        total = 0
        for unit_type, count in unit_counts.items():
            if unit_type in self.residential_units and count > 0:
                total += self.residential_units[unit_type] * count
        return total
    
    def calculate_common_load(self, installation_counts: Dict[str, int]) -> float:
        """Calculate common services load"""
        total = 0
        for install, count in installation_counts.items():
            if install in self.common_installations and count > 0:
                total += self.common_installations[install] * count
        return total
    
    def calculate_retail_load(self, shops: List[RetailShop]) -> Dict:
        """Calculate retail shops load"""
        if not shops:
            return {
                "total_kw": 0,
                "total_kva": 0,
                "shops": []
            }
        
        total_kw = sum(shop.load_kw for shop in shops if shop and hasattr(shop, 'load_kw'))
        total_kva = sum(shop.load_kva for shop in shops if shop and hasattr(shop, 'load_kva'))
        
        return {
            "total_kw": total_kw,
            "total_kva": total_kva,
            "shops": shops
        }
    
    def calculate_hawker_centre(self, area_sqm: float, cooked_food_stalls: int = 40) -> Dict:
        """Calculate hawker centre load based on typical configuration"""
        cooked_food_load = cooked_food_stalls * 8
        lighting_load = area_sqm * 0.02
        mech_vent_load = 8 * 11
        
        total_kw = cooked_food_load + lighting_load + mech_vent_load
        total_kva = total_kw / 0.85
        
        return {
            "cooked_food_stalls": cooked_food_stalls,
            "cooked_food_load_kw": cooked_food_load,
            "lighting_load_kw": lighting_load,
            "mech_vent_load_kw": mech_vent_load,
            "total_kw": total_kw,
            "total_kva": total_kva,
            "with_diversity_80%": total_kva * 0.8
        }
    
    def calculate_polyclinic(self, gfa_sqm: float) -> Dict:
        """Calculate polyclinic load"""
        normal_supply = gfa_sqm * 0.120
        emergency_supply = gfa_sqm * 0.055
        
        return {
            "gfa_sqm": gfa_sqm,
            "normal_supply_kw": normal_supply,
            "normal_supply_kva": normal_supply / 0.85,
            "emergency_supply_kw": emergency_supply,
            "emergency_supply_kva": emergency_supply / 0.85,
            "total_kva": (normal_supply + emergency_supply) / 0.85
        }

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================

def render_sidebar():
    """Render sidebar with project info"""
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100.png?text=HDB+Logo", use_column_width=True)
        st.title("Project Information")
        
        project_title = st.text_input("Project Title", "PROPOSED PUBLIC HOUSING DEVELOPMENT")
        project_ref = st.text_input("Project Reference No.", "Axxx")
        location = st.text_input("Location Description", "CCK")
        
        st.divider()
        
        st.subheader("Professional Engineer")
        pe_name = st.text_input("Name", "Ting Ik Hing")
        pe_reg_no = st.text_input("Registration No.", "3348")
        firm_name = st.text_input("Firm Name", "Surbana International Consultants Pte Ltd")
        telephone = st.text_input("Telephone No.", "62481315")
        
        st.divider()
        
        st.date_input("Date", datetime.now())
        
        return {
            "project_title": project_title,
            "project_ref": project_ref,
            "location": location,
            "pe_name": pe_name,
            "pe_reg_no": pe_reg_no,
            "firm_name": firm_name,
            "telephone": telephone
        }

def render_residential_tab(calculator):
    """Render residential units input tab"""
    st.header("🏢 Residential Units")
    
    col1, col2 = st.columns(2)
    
    unit_counts = {}
    
    with col1:
        st.subheader("Unit Types")
        for i, (unit_type, load) in enumerate(list(calculator.residential_units.items())[:3]):
            unit_counts[unit_type] = st.number_input(
                f"{unit_type} ({load} kVA)",
                min_value=0,
                value=0,
                step=1,
                key=f"res_{i}"
            )
    
    with col2:
        st.subheader("Unit Types (cont.)")
        for i, (unit_type, load) in enumerate(list(calculator.residential_units.items())[3:]):
            unit_counts[unit_type] = st.number_input(
                f"{unit_type} ({load} kVA)",
                min_value=0,
                value=0,
                step=1,
                key=f"res_{i+3}"
            )
    
    # Sample data from Excel
    with st.expander("📋 Load Sample Data"):
        if st.button("Load Sample Residential Data"):
            unit_counts = {
                "Studio Apartment (1-room)": 0,
                "Studio Apartment (2-room)": 0,
                "2-room flat": 68,
                "3-room flat": 0,
                "4-room flat": 0,
                "5-room flat": 515
            }
            st.success("Sample data loaded!")
            st.rerun()
    
    return unit_counts

def render_common_services_tab(calculator):
    """Render common services input tab"""
    st.header("⚙️ Common Services")
    
    installation_counts = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Lifts & Escalators")
        installation_counts["Escalator"] = st.number_input("Escalator (22 kVA)", 0, 10, 2)
        installation_counts["Service Lift (20 man)"] = st.number_input("Service Lift (20 man, 35 kVA)", 0, 10, 1)
        installation_counts["Lift 11 sty (13 man)"] = st.number_input("Lift 11 sty (13 man, 20 kVA)", 0, 10, 1)
        installation_counts["Lift (20 man)"] = st.number_input("Lift (20 man, 35 kVA)", 0, 10, 0)
    
    with col2:
        st.subheader("Pumps & Lighting")
        installation_counts["Domestic Booster Pump"] = st.number_input("Domestic Booster Pump (3.52 kVA)", 0, 10, 0)
        installation_counts["Fire Hose Reel Pump"] = st.number_input("Fire Hose Reel Pump (2.6 kVA)", 0, 10, 1)
        installation_counts["Public Lighting Circuit"] = st.number_input("Public Lighting Circuit (12 kVA)", 0, 10, 1)
        installation_counts["Outdoor Lighting Circuit"] = st.number_input("Outdoor Lighting Circuit (17 kVA)", 0, 10, 1)
    
    st.subheader("Fire Protection Systems")
    col3, col4 = st.columns(2)
    with col3:
        installation_counts["Sprinkler System"] = st.number_input("Sprinkler System (44 kVA)", 0, 10, 0)
        installation_counts["Wet Riser System"] = st.number_input("Wet Riser System (25 kVA)", 0, 10, 0)
    with col4:
        installation_counts["Refuse Handling Plant"] = st.number_input("Refuse Handling Plant (10 kVA)", 0, 10, 1)
        installation_counts["Mech Ventilation"] = st.number_input("Mech Ventilation (11 kVA)", 0, 20, 0)
    
    return installation_counts

def render_facilities_tab(calculator):
    """Render facilities input tab"""
    st.header("🏛️ Facilities")
    
    col1, col2 = st.columns(2)
    
    facility_loads = {}
    
    with col1:
        st.subheader("Community Facilities")
        facility_loads["Future Communal Facilities"] = st.number_input("Future Communal Facilities (42 kVA)", 0, 1000, 0)
        facility_loads["Bin Centre"] = st.number_input("Bin Centre (10 kVA)", 0, 100, 0)
        facility_loads["RC Centre"] = st.number_input("RC Centre (42 kVA)", 0, 100, 0)
        facility_loads["CC"] = st.number_input("CC (693 kVA)", 0, 2000, 0)
        
        st.subheader("Retail")
        num_shops = st.number_input("Number of Retail Shops", 0, 20, 6)
    
    with col2:
        st.subheader("Healthcare")
        facility_loads["Polyclinic"] = st.number_input("Polyclinic (881.28 kVA)", 0, 2000, 0)
        facility_loads["KDC"] = st.number_input("KDC (104 kVA)", 0, 500, 0)
        
        st.subheader("Others")
        facility_loads["Hawker Centre"] = st.number_input("Hawker Centre (676.88 kVA/3500sqm)", 0, 2000, 0)
        facility_loads["MNO"] = st.number_input("MNO (41.6 kVA)", 0, 500, 0)
        facility_loads["EPS"] = st.number_input("EPS (14 kVA)", 0, 500, 0)
        facility_loads["Mech"] = st.number_input("Mech (762 kVA)", 0, 2000, 0)
    
    return facility_loads, num_shops

def render_retail_shops():
    """Render retail shops input"""
    st.header("🛍️ Retail Shops Details")
    
    # Sample shops from Excel
    sample_shops = [
        {"name": "Retail (Chinese medicine)", "area": 86.3, "breaker": "63A SPN", "load_kw": 14.49},
        {"name": "Convenience Store", "area": 121.28, "breaker": "40A TPN", "load_kw": 27.712},
        {"name": "Open Retail", "area": 27.98, "breaker": "63A SPN", "load_kw": 14.49},
        {"name": "Halal Café", "area": 121.34, "breaker": "40A TPN", "load_kw": 27.712},
        {"name": "Takeaway food", "area": 50.85, "breaker": "63A SPN", "load_kw": 14.49},
        {"name": "Gym", "area": 299.93, "breaker": "80A TPN", "load_kw": 55.424}
    ]
    
    shops = []
    
    # Initialize session state for shops if not exists
    if 'shops' not in st.session_state:
        st.session_state.shops = sample_shops.copy()
    
    # Display shops in a dataframe for editing
    df_shops = pd.DataFrame(st.session_state.shops)
    
    # Rename columns for display
    display_df = df_shops.rename(columns={
        "name": "Shop Name",
        "area": "Area (m²)",
        "breaker": "Breaker Size",
        "load_kw": "Load (kW)"
    })
    
    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Shop Name": st.column_config.TextColumn("Shop Name", required=True),
            "Area (m²)": st.column_config.NumberColumn("Area (m²)", min_value=0, format="%.2f", required=True),
            "Breaker Size": st.column_config.TextColumn("Breaker Size", required=True),
            "Load (kW)": st.column_config.NumberColumn("Load (kW)", min_value=0, format="%.3f", required=True)
        }
    )
    
    # Convert back to original column names and update session state
    if not edited_df.empty:
        updated_shops = []
        for _, row in edited_df.iterrows():
            shop_dict = {
                "name": row["Shop Name"],
                "area": row["Area (m²)"],
                "breaker": row["Breaker Size"],
                "load_kw": row["Load (kW)"]
            }
            updated_shops.append(shop_dict)
        
        st.session_state.shops = updated_shops
        
        # Convert to RetailShop objects
        try:
            shops = [RetailShop(
                name=shop["name"],
                area_sqm=shop["area"],
                breaker_size=shop["breaker"],
                load_kw=shop["load_kw"],
                power_factor=0.85
            ) for shop in st.session_state.shops]
        except Exception as e:
            st.error(f"Error creating shop objects: {str(e)}")
            shops = []
    
    return shops

def render_hawker_centre_detail():
    """Render detailed hawker centre calculation"""
    st.header("🍜 Hawker Centre Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("Area (sqm)", 0, 10000, 3500)
        cooked_food_stalls = st.number_input("Number of Cooked Food Stalls", 0, 100, 40)
        lighting_density = st.number_input("Lighting Density (W/m²)", 0.0, 50.0, 20.0)
    
    with col2:
        mv_motors = st.number_input("Number of MV Motors", 0, 20, 8)
        mv_motor_power = st.number_input("MV Motor Power (kW)", 0.0, 50.0, 11.0)
        management_office_ac = st.number_input("Management Office AC (kW)", 0.0, 20.0, 3.0)
    
    # Calculate loads
    cooked_food_load = cooked_food_stalls * 8
    lighting_load = area * lighting_density / 1000
    mv_load = mv_motors * mv_motor_power
    total_kw = cooked_food_load + lighting_load + mv_load + management_office_ac
    total_kva = total_kw / 0.85
    
    # Display results
    st.subheader("Hawker Centre Load Summary")
    results_df = pd.DataFrame({
        "Description": ["Cooked Food Stalls", "Lighting", "MV Motors", "Management Office AC", "TOTAL"],
        "Load (kW)": [cooked_food_load, lighting_load, mv_load, management_office_ac, total_kw],
        "Load (kVA)": [cooked_food_load/0.85, lighting_load/0.85, mv_load/0.85, management_office_ac/0.85, total_kva]
    })
    st.dataframe(results_df, use_container_width=True)
    
    return {
        "area": area,
        "cooked_food_stalls": cooked_food_stalls,
        "total_kw": total_kw,
        "total_kva": total_kva
    }

def render_distribution_system():
    """Render distribution system diagram"""
    st.header("🔌 Distribution System")
    
    # Distribution data from Excel
    distribution_data = {
        "PG Incoming": ["PG Incoming 1", "PG Incoming 2", "PG Incoming 3", "PG Incoming 4"],
        "MSB": ["MSB 1", "MSB 1", "MSB 2", "MSB 2"],
        "SSB": ["SSB-Resi", "SSB-PC", "SSB-CC", "SSB-HWC"],
        "Other": ["SSB-Shop, Chiller Plant", "SSB-KDC", "EMSB-1", "EMSB-2"],
        "Load (kVA)": [1067, 985, 951, 925]
    }
    
    df_dist = pd.DataFrame(distribution_data)
    st.dataframe(df_dist, use_container_width=True)
    
    # Create a simple distribution diagram
    fig = go.Figure()
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=[0, 1, 2, 3, 4],
        y=[2, 2, 2, 2, 2],
        mode='markers+text',
        marker=dict(size=30, color='lightblue'),
        text=['Source', 'PG1', 'PG2', 'PG3', 'PG4'],
        textposition="middle center",
        name=''
    ))
    
    fig.update_layout(
        title="Single Line Diagram (Simplified)",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_results(calculator, unit_counts, installation_counts, facility_loads, shops, hawker_detail):
    """Render calculation results"""
    st.header("📊 Electrical Load Summary")
    
    # Calculate loads
    residential_load = calculator.calculate_residential_load(unit_counts)
    common_load = calculator.calculate_common_load(installation_counts)
    
    # Facilities load - ensure we're summing only numeric values
    facilities_total = 0
    for facility, load in facility_loads.items():
        try:
            facilities_total += float(load) if load else 0
        except (ValueError, TypeError):
            pass
    
    # Retail load
    retail_result = {"total_kva": 0, "total_kw": 0, "shops": []}
    if shops:
        retail_result = calculator.calculate_retail_load(shops)
    
    # Calculate total unmetered and metered
    unmetered_load = residential_load + common_load + facilities_total
    metered_load = retail_result['total_kva']
    
    total_load = unmetered_load + metered_load
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Residential Load", f"{residential_load:.2f} kVA")
        st.metric("Common Services", f"{common_load:.2f} kVA")
    
    with col2:
        st.metric("Facilities", f"{facilities_total:.2f} kVA")
        st.metric("Retail", f"{retail_result['total_kva']:.2f} kVA")
    
    with col3:
        st.metric("Unmetered Supply", f"{unmetered_load:.2f} kVA")
        st.metric("Metered Supply", f"{metered_load:.2f} kVA")
        st.metric("**TOTAL LOAD**", f"**{total_load:.2f} kVA**", delta=f"{total_load/1000:.2f} MVA")
    
    # Create detailed breakdown
    st.subheader("Detailed Load Breakdown")
    
    # Prepare data for visualization
    breakdown_data = []
    
    # Residential breakdown
    for unit_type, count in unit_counts.items():
        if count > 0:
            breakdown_data.append({
                "Category": "Residential",
                "Item": unit_type,
                "Load (kVA)": calculator.residential_units[unit_type] * count
            })
    
    # Common services breakdown
    for install, count in installation_counts.items():
        if count > 0 and install in calculator.common_installations:
            breakdown_data.append({
                "Category": "Common Services",
                "Item": install,
                "Load (kVA)": calculator.common_installations[install] * count
            })
    
    # Facilities breakdown
    for facility, load in facility_loads.items():
        if load and load > 0:
            breakdown_data.append({
                "Category": "Facilities",
                "Item": facility,
                "Load (kVA)": load
            })
    
    # Retail breakdown
    for shop in shops:
        if shop and hasattr(shop, 'name'):
            breakdown_data.append({
                "Category": "Retail",
                "Item": shop.name,
                "Load (kVA)": shop.load_kva
            })
    
    if breakdown_data:
        df_breakdown = pd.DataFrame(breakdown_data)
        
        # Pie chart
        fig = px.pie(df_breakdown, values='Load (kVA)', names='Category', 
                     title='Load Distribution by Category')
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart
        fig2 = px.bar(df_breakdown, x='Item', y='Load (kVA)', color='Category',
                      title='Individual Loads')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Table
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
    
    # Transformer sizing
    st.subheader("🔋 Transformer Sizing")
    
    transformer_capacity = total_load * 1.1
    num_transformers = max(1, int(np.ceil(transformer_capacity / 1000)))
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Recommended: **{num_transformers} x 1MVA Transformers**")
        st.info(f"Total Capacity: **{num_transformers * 1000:.0f} kVA**")
    
    with col2:
        if transformer_capacity > num_transformers * 1000:
            st.warning(f"⚠️ Load exceeds transformer capacity by {transformer_capacity - num_transformers * 1000:.0f} kVA")
        else:
            st.success(f"✅ Adequate capacity (Margin: {num_transformers * 1000 - transformer_capacity:.0f} kVA)")
    
    # Generator sizing
    st.subheader("⚡ Standby Generator Sizing")
    emergency_load = common_load * 0.3 + facility_loads.get("Polyclinic", 0) * 0.4
    generator_size = max(700, int(np.ceil(emergency_load / 100) * 100))
    
    st.info(f"Recommended: **1 x {generator_size} kVA Generator**")
    
    # Professional declaration
    st.subheader("📝 Professional Engineer's Declaration")
    
    st.markdown("""
    I, the Professional Engineer for the declared Electrical Works, hereby submit the electrical design load data and calculation and confirm that the details given in this form are to the best of my knowledge true and correct.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("PE Name", "Ting Ik Hing", key="pe_name_result")
        st.text_input("Registration No.", "3348", key="pe_reg_result")
    with col2:
        st.text_input("Firm Name", "Surbana International Consultants Pte Ltd", key="firm_result")
        st.text_input("Date", datetime.now().strftime("%Y-%m-%d"), key="date_result")

def render_export_options():
    """Render export options"""
    st.header("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export to Excel", use_container_width=True):
            st.success("Excel export functionality - would generate file")
    
    with col2:
        if st.button("📊 Export to PDF", use_container_width=True):
            st.success("PDF export functionality - would generate report")
    
    with col3:
        if st.button("💾 Save Project", use_container_width=True):
            st.success("Project saved to session")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="HDB Electrical Load Calculator",
        page_icon="⚡",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7f9;
    }
    .main-header {
        background-color: #003366;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header"><h1>⚡ HDB Electrical Design Load Calculator</h1><p>Professional Edition - CCK Project Template</p></div>', 
                unsafe_allow_html=True)
    
    # Initialize calculator
    calculator = ElectricalLoadCalculator()
    
    # Sidebar for project info
    project_info = render_sidebar()
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏢 Residential", "⚙️ Common Services", "🏛️ Facilities", 
        "🍜 Hawker Centre", "🔌 Distribution", "📊 Results"
    ])
    
    with tab1:
        unit_counts = render_residential_tab(calculator)
    
    with tab2:
        installation_counts = render_common_services_tab(calculator)
    
    with tab3:
        facility_loads, num_shops = render_facilities_tab(calculator)
        shops = render_retail_shops()
    
    with tab4:
        hawker_detail = render_hawker_centre_detail()
    
    with tab5:
        render_distribution_system()
    
    with tab6:
        render_results(calculator, unit_counts, installation_counts, facility_loads, shops, hawker_detail)
        render_export_options()
    
    # Footer
    st.divider()
    st.caption(f"© 2024 HDB - Electrical Design Load Calculator v2.0 | Project: {project_info['project_title']} | Reference: {project_info['project_ref']}")

if __name__ == "__main__":
    main()
