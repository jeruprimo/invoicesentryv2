import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Set
import sqlite3
import re

def init_db():
    """Initialize database with proper error handling"""
    try:
        conn = sqlite3.connect('invoice_sentry.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS invoices (
                trip_id TEXT,
                vendor_name TEXT,
                vendor_email TEXT,
                amount REAL,
                days_overdue INTEGER,
                status TEXT DEFAULT 'Pending',
                UNIQUE(trip_id, vendor_email)
            )
        ''')
        # Create table for tracking sent requests
        c.execute('''
            CREATE TABLE IF NOT EXISTS sent_requests (
                trip_id TEXT,
                vendor_email TEXT,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (trip_id, vendor_email)
            )
        ''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        st.error(f"Database initialization error: {e}")

def load_sent_requests() -> Set[str]:
    """Load sent requests from database"""
    try:
        conn = sqlite3.connect('invoice_sentry.db')
        c = conn.cursor()
        c.execute('SELECT trip_id, vendor_email FROM sent_requests')
        rows = c.fetchall()
        conn.close()
        return {f"{row[0]}_{row[1]}" for row in rows}
    except sqlite3.Error as e:
        st.warning(f"Could not load sent requests from database: {e}")
        return set()

def save_sent_request(trip_id: str, vendor_email: str):
    """Save sent request to database and update status in invoices table"""
    try:
        conn = sqlite3.connect('invoice_sentry.db')
        c = conn.cursor()
        # Save to sent_requests table
        c.execute('''
            INSERT OR IGNORE INTO sent_requests (trip_id, vendor_email)
            VALUES (?, ?)
        ''', (trip_id, vendor_email))
        # Update status in invoices table
        c.execute('''
            UPDATE invoices
            SET status = 'Sent'
            WHERE trip_id = ? AND vendor_email = ?
        ''', (trip_id, vendor_email))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        st.warning(f"Could not save sent request to database: {e}")

def save_invoices_to_db(df: pd.DataFrame):
    """Save invoice data to database using INSERT OR REPLACE, preserving existing status"""
    try:
        conn = sqlite3.connect('invoice_sentry.db')
        c = conn.cursor()
        # Map DataFrame columns to database columns
        for _, row in df.iterrows():
            # Check if record exists to preserve status
            c.execute('SELECT status FROM invoices WHERE trip_id = ? AND vendor_email = ?', 
                     (row['Trip ID'], row['Vendor Email']))
            existing = c.fetchone()
            status = existing[0] if existing else 'Pending'
            
            # Insert or replace with preserved status
            c.execute('''
                INSERT OR REPLACE INTO invoices 
                (trip_id, vendor_name, vendor_email, amount, days_overdue, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                row['Trip ID'],
                row['Vendor Name'],
                row['Vendor Email'],
                row['Amount'],
                row['Days Overdue'],
                status
            ))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"Could not save invoices to database: {e}")
        return False

def load_invoices_from_db() -> pd.DataFrame:
    """Load invoice data from database"""
    try:
        conn = sqlite3.connect('invoice_sentry.db')
        df = pd.read_sql_query('''
            SELECT 
                trip_id AS "Trip ID",
                vendor_name AS "Vendor Name",
                vendor_email AS "Vendor Email",
                amount AS "Amount",
                days_overdue AS "Days Overdue",
                status AS "Request Status"
            FROM invoices
            ORDER BY trip_id, vendor_name
        ''', conn)
        conn.close()
        # Ensure Days Overdue is integer
        if not df.empty:
            df['Days Overdue'] = df['Days Overdue'].astype(int)
            # Ensure Request Status has default value if None
            if 'Request Status' in df.columns:
                df['Request Status'] = df['Request Status'].fillna('Pending')
        return df
    except sqlite3.Error as e:
        st.warning(f"Could not load invoices from database: {e}")
        return pd.DataFrame()
    except pd.errors.DatabaseError as e:
        st.warning(f"Database query error: {e}")
        return pd.DataFrame()

def has_invoices_in_db() -> bool:
    """Check if there are any invoices in the database"""
    try:
        conn = sqlite3.connect('invoice_sentry.db')
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM invoices')
        count = c.fetchone()[0]
        conn.close()
        return count > 0
    except sqlite3.Error:
        return False

def validate_email(email: str) -> bool:
    """Basic email validation"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

# Run initialization
init_db()

# Page configuration
st.set_page_config(
    page_title="Invoice Sentry Dashboard",
    page_icon="📋",
    layout="wide"
)

# Initialize session state for tracking sent requests
# Load from database on first run
if 'sent_requests' not in st.session_state:
    st.session_state.sent_requests = load_sent_requests()

if 'preview_email' not in st.session_state:
    st.session_state.preview_email = None

if 'df' not in st.session_state:
    st.session_state.df = None


# Sample data structure
@st.cache_data
def get_sample_data():
    """Generate sample invoice data grouped by Trip Number"""
    data = [
        {
            'Trip ID': 'TRIP-001',
            'Vendor Name': 'ABC Logistics',
            'Vendor Email': 'invoices@abclogistics.com',
            'Amount': 12500.00,
            'Days Overdue': 15,
            'Invoice Date': datetime.now() - timedelta(days=45)
        },
        {
            'Trip ID': 'TRIP-001',
            'Vendor Name': 'XYZ Transport',
            'Vendor Email': 'billing@xyztransport.com',
            'Amount': 8500.00,
            'Days Overdue': 8,
            'Invoice Date': datetime.now() - timedelta(days=38)
        },
        {
            'Trip ID': 'TRIP-002',
            'Vendor Name': 'Global Shipping Co.',
            'Vendor Email': 'accounts@globalshipping.com',
            'Amount': 22000.00,
            'Days Overdue': 22,
            'Invoice Date': datetime.now() - timedelta(days=52)
        },
        {
            'Trip ID': 'TRIP-002',
            'Vendor Name': 'Fast Freight Inc.',
            'Vendor Email': 'finance@fastfreight.com',
            'Amount': 15000.00,
            'Days Overdue': 12,
            'Invoice Date': datetime.now() - timedelta(days=42)
        },
        {
            'Trip ID': 'TRIP-002',
            'Vendor Name': 'Premium Logistics',
            'Vendor Email': 'invoices@premiumlogistics.com',
            'Amount': 9800.00,
            'Days Overdue': 5,
            'Invoice Date': datetime.now() - timedelta(days=35)
        },
        {
            'Trip ID': 'TRIP-003',
            'Vendor Name': 'Express Delivery',
            'Vendor Email': 'billing@expressdelivery.com',
            'Amount': 11200.00,
            'Days Overdue': 18,
            'Invoice Date': datetime.now() - timedelta(days=48)
        },
        {
            'Trip ID': 'TRIP-004',
            'Vendor Name': 'Critical Freight',
            'Vendor Email': 'billing@criticalfreight.com',
            'Amount': 35000.00,
            'Days Overdue': 25,
            'Invoice Date': datetime.now() - timedelta(days=55)
        },
        {
            'Trip ID': 'TRIP-004',
            'Vendor Name': 'Urgent Logistics',
            'Vendor Email': 'accounts@urgentlogistics.com',
            'Amount': 18000.00,
            'Days Overdue': 23,
            'Invoice Date': datetime.now() - timedelta(days=53)
        },
        {
            'Trip ID': 'TRIP-005',
            'Vendor Name': 'Standard Shipping',
            'Vendor Email': 'invoices@standardshipping.com',
            'Amount': 7500.00,
            'Days Overdue': 11,
            'Invoice Date': datetime.now() - timedelta(days=41)
        },
        {
            'Trip ID': 'TRIP-005',
            'Vendor Name': 'Quick Transport',
            'Vendor Email': 'billing@quicktransport.com',
            'Amount': 9200.00,
            'Days Overdue': 9,
            'Invoice Date': datetime.now() - timedelta(days=39)
        },
    ]
    return pd.DataFrame(data)

def generate_email_preview(trip_id: str, vendor_name: str, vendor_email: str, amount: float, days_overdue: int) -> str:
    """Generate email template preview for a vendor"""
    return f"""
Subject: Request for Missing Invoice - {trip_id}

Dear {vendor_name},

We are writing to request a copy of the missing invoice for Trip ID: {trip_id}.

Details:
- Trip ID: {trip_id}
- Amount: ${amount:,.2f}
- Days Overdue: {days_overdue} days

Please send the invoice to our accounts payable department at your earliest convenience.

Thank you for your prompt attention to this matter.

Best regards,
Accounts Payable Team
"""

def send_request(trip_id: str, vendor_email: str):
    """Simulate sending a request (placeholder for future email logic)"""
    request_key = f"{trip_id}_{vendor_email}"
    st.session_state.sent_requests.add(request_key)
    # Persist to database
    save_sent_request(trip_id, vendor_email)
    # Update session state dataframe if it exists
    if st.session_state.df is not None and 'Request Status' in st.session_state.df.columns:
        mask = (st.session_state.df['Trip ID'] == trip_id) & (st.session_state.df['Vendor Email'] == vendor_email)
        st.session_state.df.loc[mask, 'Request Status'] = 'Sent'

def send_bulk_requests(trip_id: str, vendors: pd.DataFrame):
    """Send requests for all vendors in a trip"""
    for _, vendor in vendors.iterrows():
        request_key = f"{trip_id}_{vendor['Vendor Email']}"
        if request_key not in st.session_state.sent_requests:
            send_request(trip_id, vendor['Vendor Email'])

def get_status_color(days_overdue: int) -> str:
    """Get status color indicator based on days overdue"""
    if days_overdue > 20:
        return "🔴"
    elif days_overdue > 10:
        return "🟡"
    else:
        return "🟢"

def get_status_category(days_overdue: int) -> str:
    """Get status category: Red, Yellow, or Green"""
    if days_overdue > 20:
        return "Red"
    elif days_overdue > 10:
        return "Yellow"
    else:
        return "Green"

def filter_and_sort_trips(df: pd.DataFrame, search_query: str, status_filter: str, view_type: str) -> List[Tuple[str, pd.DataFrame, float]]:
    """
    Filter and sort trips based on search, status filter, and view type.
    Returns list of tuples: (trip_id, trip_vendors_df, max_days_overdue)
    """
    # Apply search filter
    if search_query:
        search_lower = search_query.lower()
        mask = (
            df['Trip ID'].str.lower().str.contains(search_lower, na=False) |
            df['Vendor Name'].str.lower().str.contains(search_lower, na=False) |
            df['Vendor Email'].str.lower().str.contains(search_lower, na=False)
        )
        filtered_df = df[mask].copy()
    else:
        filtered_df = df.copy()
    
    # Add urgency status category column (Red/Yellow/Green based on days overdue)
    filtered_df['Urgency Status'] = filtered_df['Days Overdue'].apply(get_status_category)
    
    # Apply status filter (based on urgency)
    if status_filter == "Red Only":
        filtered_df = filtered_df[filtered_df['Urgency Status'] == 'Red'].copy()
    elif status_filter == "Yellow Only":
        filtered_df = filtered_df[filtered_df['Urgency Status'] == 'Yellow'].copy()
    # "Show All" doesn't filter
    
    # Add request status column
    filtered_df['Request Key'] = filtered_df.apply(
        lambda row: f"{row['Trip ID']}_{row['Vendor Email']}", axis=1
    )
    # Check both session state and database request status column
    # Get request status from database if available, otherwise check session state
    if 'Request Status' in filtered_df.columns:
        filtered_df['Is Sent'] = (
            filtered_df['Request Key'].isin(st.session_state.sent_requests) |
            (filtered_df['Request Status'] == 'Sent')
        )
    else:
        filtered_df['Is Sent'] = filtered_df['Request Key'].isin(st.session_state.sent_requests)
    
    # Apply view type filter
    if view_type == "Needs Action":
        # Only show vendors that haven't been requested
        filtered_df = filtered_df[~filtered_df['Is Sent']].copy()
    elif view_type == "Awaiting Invoice":
        # Only show vendors that have been requested
        filtered_df = filtered_df[filtered_df['Is Sent']].copy()
    # "All Trips" shows everything
    
    # Group by Trip ID and calculate max days overdue for sorting
    trip_groups = []
    if len(filtered_df) > 0:
        for trip_id in filtered_df['Trip ID'].unique():
            trip_vendors = filtered_df[filtered_df['Trip ID'] == trip_id].copy()
            max_days = trip_vendors['Days Overdue'].max()
            has_red = (trip_vendors['Urgency Status'] == 'Red').any()
            has_yellow = (trip_vendors['Urgency Status'] == 'Yellow').any()
            
            trip_groups.append((trip_id, trip_vendors, max_days, has_red, has_yellow))
    
    # Smart sorting: Red first, then by max days overdue (descending)
    trip_groups.sort(key=lambda x: (
        not x[3],  # Red trips first (True sorts before False)
        not x[4],  # Yellow trips second
        -x[2]      # Then by max days overdue (descending)
    ))
    
    return [(trip_id, vendors, max_days) for trip_id, vendors, max_days, _, _ in trip_groups]

def send_all_red_requests(df: pd.DataFrame):
    """Send requests for all Red (overdue > 20 days) vendors across all trips"""
    red_vendors = df[df['Days Overdue'] > 20]
    count = 0
    for _, vendor in red_vendors.iterrows():
        request_key = f"{vendor['Trip ID']}_{vendor['Vendor Email']}"
        if request_key not in st.session_state.sent_requests:
            send_request(vendor['Trip ID'], vendor['Vendor Email'])
            count += 1
    return count

def load_csv_data(uploaded_file) -> pd.DataFrame:
    """Load and validate CSV data"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Required columns check
        required_columns = ['Trip ID', 'Vendor Name', 'Vendor Email', 'Amount', 'Days Overdue']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info("Please ensure your CSV has: Trip ID, Vendor Name, Vendor Email, Amount, Days Overdue")
            return None
        
        # Validate email addresses
        invalid_emails = df[~df['Vendor Email'].apply(validate_email)]
        if len(invalid_emails) > 0:
            st.warning(f"⚠️ Found {len(invalid_emails)} invalid email address(es). They will be excluded.")
            invalid_email_list = invalid_emails[['Trip ID', 'Vendor Email']].to_dict('records')
            for item in invalid_email_list[:5]:  # Show first 5
                st.caption(f"  - {item['Trip ID']}: {item['Vendor Email']}")
            if len(invalid_emails) > 5:
                st.caption(f"  ... and {len(invalid_emails) - 5} more")
            df = df[df['Vendor Email'].apply(validate_email)]
        
        # Ensure Days Overdue is numeric
        df['Days Overdue'] = pd.to_numeric(df['Days Overdue'], errors='coerce')
        
        # Check for any NaN values in Days Overdue
        if df['Days Overdue'].isna().any():
            st.warning("⚠️ Some 'Days Overdue' values could not be converted to numbers. They will be excluded.")
            df = df.dropna(subset=['Days Overdue'])
        
        # Ensure Amount is numeric
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        if df['Amount'].isna().any():
            st.warning("⚠️ Some 'Amount' values could not be converted to numbers. They will be excluded.")
            df = df.dropna(subset=['Amount'])
        
        # Convert Days Overdue to integer
        df['Days Overdue'] = df['Days Overdue'].astype(int)
        
        return df
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        return None

def create_template_csv() -> bytes:
    """Create a template CSV file for download"""
    template_data = {
        'Trip ID': ['TRIP-001', 'TRIP-001', 'TRIP-002'],
        'Vendor Name': ['ABC Logistics', 'XYZ Transport', 'Global Shipping Co.'],
        'Vendor Email': ['invoices@abclogistics.com', 'billing@xyztransport.com', 'accounts@globalshipping.com'],
        'Amount': [12500.00, 8500.00, 22000.00],
        'Days Overdue': [15, 8, 22]
    }
    df_template = pd.DataFrame(template_data)
    return df_template.to_csv(index=False).encode('utf-8')

def render_trip_expander(trip_id: str, trip_vendors: pd.DataFrame, max_days: float):
    """Render a single trip expander with all its vendors"""
    with st.expander(f"🚛 {trip_id} - {len(trip_vendors)} Vendor(s)", expanded=False):
        # Bulk action button at the top
        col_bulk, col_info = st.columns([1, 3])
        with col_bulk:
            if st.button(
                "📧 Request All Missing Invoices for this Trip",
                key=f"bulk_{trip_id}",
                type="primary"
            ):
                send_bulk_requests(trip_id, trip_vendors)
                st.rerun()
        
        with col_info:
            total_amount = trip_vendors['Amount'].sum()
            st.caption(f"Total Amount: ${total_amount:,.2f} | Max Days Overdue: {int(max_days)}")
        
        st.divider()
        
        # Display vendor rows
        for idx, vendor in trip_vendors.iterrows():
            request_key = f"{trip_id}_{vendor['Vendor Email']}"
            is_sent = request_key in st.session_state.sent_requests
            
            # Create columns for vendor information
            col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 1.5, 1, 1.5])
            
            with col1:
                st.write(f"**{vendor['Vendor Name']}**")
                st.caption(vendor['Vendor Email'])
            
            with col2:
                st.metric("Amount", f"${vendor['Amount']:,.2f}")
            
            with col3:
                # Color code days overdue
                days = vendor['Days Overdue']
                color = get_status_color(days)
                st.metric("Days Overdue", f"{color} {days}")
            
            with col4:
                # Email preview button
                email_preview = generate_email_preview(
                    trip_id,
                    vendor['Vendor Name'],
                    vendor['Vendor Email'],
                    vendor['Amount'],
                    vendor['Days Overdue']
                )
                if st.button(
                    "👁️ Preview",
                    key=f"preview_{trip_id}_{idx}",
                    help="Click to see email preview",
                    use_container_width=True
                ):
                    st.session_state.preview_email = {
                        'vendor': vendor['Vendor Name'],
                        'content': email_preview
                    }
                    st.rerun()
            
            with col5:
                # Send Request button
                if is_sent:
                    st.success("✅ Sent", use_container_width=True)
                else:
                    if st.button(
                        "📤 Send Request",
                        key=f"send_{trip_id}_{idx}",
                        use_container_width=True
                    ):
                        send_request(trip_id, vendor['Vendor Email'])
                        st.rerun()
            
            with col6:
                # Status indicator
                if is_sent:
                    st.caption("Status: Sent")
                else:
                    st.caption("Status: Pending")
            
            # Add spacing between vendor rows
            st.markdown("<br>", unsafe_allow_html=True)

# Main app
def main():
    # Load data from database at the very beginning if not already loaded
    # Always reload from database to ensure we have the latest status
    df_from_db = load_invoices_from_db()
    if not df_from_db.empty:
        st.session_state.df = df_from_db
    elif 'df' not in st.session_state:
        st.session_state.df = None
    
    st.title("📋 Invoice Sentry Dashboard")
    st.markdown("Automate the request process for missing invoices and track their arrival")
    
    # Sidebar with filters and file upload
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # CSV file uploader
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file with columns: Trip ID, Vendor Name, Vendor Email, Amount, Days Overdue"
        )
        
        if uploaded_file is not None:
            # Load and validate CSV data
            df_uploaded = load_csv_data(uploaded_file)
            if df_uploaded is not None:
                # Save to database immediately
                if save_invoices_to_db(df_uploaded):
                    st.session_state.df = df_uploaded
                    st.success(f"✅ Loaded and saved {len(df_uploaded)} vendor records from {uploaded_file.name} to database")
                else:
                    st.error("❌ Failed to save data to database")
            else:
                # Keep existing data if upload failed
                if st.session_state.df is None:
                    st.info("Please fix CSV format and try again.")
        else:
            # Show clear button if data is uploaded
            if st.session_state.df is not None:
                if st.button("Clear Uploaded Data"):
                    st.session_state.df = None
                    st.rerun()
        
        # Download template CSV button
        st.download_button(
            label="📥 Download Template CSV",
            data=create_template_csv(),
            file_name="invoice_sentry_template.csv",
            mime="text/csv",
            help="Download a template CSV file with the correct format"
        )
        
        st.divider()
        st.header("🔍 Filters")
        
        # Search bar
        search_query = st.text_input(
            "Search",
            placeholder="Trip ID or Vendor Name...",
            help="Search by Trip ID, Vendor Name, or Vendor Email"
        )
        
        # Status filter
        status_filter = st.selectbox(
            "Status Filter",
            ["Show All", "Red Only", "Yellow Only"],
            help="Filter by urgency status"
        )
        
        st.divider()
        
        # Nuclear Button - Request All Red Overdue Invoices
        st.header("⚡ Bulk Actions")
        
        # Load data: use session state data if available, otherwise load from database
        if st.session_state.df is not None:
            df_sidebar = st.session_state.df.copy()
        else:
            df_sidebar = load_invoices_from_db()
            if df_sidebar.empty:
                # Only use sample data if database is completely empty
                if not has_invoices_in_db():
                    df_sidebar = get_sample_data()
        
        red_not_sent = 0
        for _, vendor in df_sidebar[df_sidebar['Days Overdue'] > 20].iterrows():
            request_key = f"{vendor['Trip ID']}_{vendor['Vendor Email']}"
            if request_key not in st.session_state.sent_requests:
                red_not_sent += 1
        
        if red_not_sent > 0:
            if st.button(
                f"🚨 Request All Overdue Invoices ({red_not_sent} Red)",
                type="primary",
                use_container_width=True,
                help="Send requests to all vendors with >20 days overdue across all trips"
            ):
                count = send_all_red_requests(df_sidebar)
                st.success(f"✅ Sent {count} requests!")
                st.rerun()
        else:
            st.info("✅ All red overdue invoices have been requested")
        
        st.divider()
        
        # Summary stats in sidebar
        st.header("📊 Quick Stats")
        total_trips = len(df_sidebar['Trip ID'].unique())
        total_vendors = len(df_sidebar)
        sent_count = len(st.session_state.sent_requests)
        red_count = len(df_sidebar[df_sidebar['Days Overdue'] > 20])
        yellow_count = len(df_sidebar[(df_sidebar['Days Overdue'] > 10) & (df_sidebar['Days Overdue'] <= 20)])
        
        st.metric("Total Trips", total_trips)
        st.metric("Total Vendors", total_vendors)
        st.metric("Requests Sent", f"{sent_count}/{total_vendors}")
        st.metric("🔴 Red (Critical)", red_count)
        st.metric("🟡 Yellow (Warning)", yellow_count)
    
    # Load data for main content: use session state data if available, otherwise load from database
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
    else:
        df = load_invoices_from_db()
        if df.empty:
            # Only use sample data if database is completely empty
            if not has_invoices_in_db():
                df = get_sample_data()
            else:
                # Database exists but query returned empty (shouldn't happen, but handle gracefully)
                st.warning("⚠️ Database exists but no data could be loaded. Please upload a CSV file.")
                df = pd.DataFrame()
    
    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs(["🚨 Needs Action", "⏳ Awaiting Invoice", "📋 All Trips"])
    
    # Filter and sort trips
    needs_action_trips = filter_and_sort_trips(df, search_query, status_filter, "Needs Action")
    awaiting_trips = filter_and_sort_trips(df, search_query, status_filter, "Awaiting Invoice")
    all_trips = filter_and_sort_trips(df, search_query, status_filter, "All Trips")
    
    # Tab 1: Needs Action (Not requested)
    with tab1:
        if needs_action_trips:
            st.subheader(f"Trips Requiring Action ({len(needs_action_trips)} trips)")
            for trip_id, trip_vendors, max_days in needs_action_trips:
                render_trip_expander(trip_id, trip_vendors, max_days)
        else:
            st.info("🎉 No trips need action! All requests have been sent.")
    
    # Tab 2: Awaiting Invoice (Requested but not received)
    with tab2:
        if awaiting_trips:
            st.subheader(f"Trips Awaiting Invoices ({len(awaiting_trips)} trips)")
            for trip_id, trip_vendors, max_days in awaiting_trips:
                render_trip_expander(trip_id, trip_vendors, max_days)
        else:
            st.info("No trips are currently awaiting invoices.")
    
    # Tab 3: All Trips
    with tab3:
        if all_trips:
            st.subheader(f"All Trips ({len(all_trips)} trips)")
            for trip_id, trip_vendors, max_days in all_trips:
                render_trip_expander(trip_id, trip_vendors, max_days)
        else:
            st.info("No trips match the current filters.")
    
    # Display email preview if one was requested
    if st.session_state.preview_email:
        st.divider()
        with st.expander("📧 Email Preview", expanded=True):
            st.markdown(f"**Email Preview for {st.session_state.preview_email['vendor']}:**")
            st.code(st.session_state.preview_email['content'], language=None)
            if st.button("Close Preview"):
                st.session_state.preview_email = None
                st.rerun()
    
    # Summary statistics at bottom
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trips", len(df['Trip ID'].unique()))
    
    with col2:
        st.metric("Total Vendors", len(df))
    
    with col3:
        st.metric("Total Amount", f"${df['Amount'].sum():,.2f}")
    
    with col4:
        sent_count = len(st.session_state.sent_requests)
        st.metric("Requests Sent", f"{sent_count}/{len(df)}")

if __name__ == "__main__":
    main()

