import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Rainbow Agents Survey Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# SANKEY COLOR MAPPING
# =====================================================


# =====================================================
# UNIFIED COLOR SYSTEM
# =====================================================

# Your existing blue-yellow foundation (keep as-is)
LIKERT_COLORS = {'YES!': '#0068c9', 'yes': '#83c8fe', 'no': '#FFEAB0', 'NO!': '#ffc932'}

# Cherry-picked from your Sankey palette
SELECTIVE_COLORS = {
    'purple': '#ba96e2',    # Netherlands - collaboration
    'teal': '#1DB3B9',      # Canada - alternatives  
    'orange': '#FFC107',    # Italy - emphasis
    'green': '#4CAF50',     # Mexico - success
    'gray': '#98BACF'       # neutral
}

# Color mappings by data type
DEMOGRAPHIC_COLORS = {
    'male': '#0068c9',           # Your primary blue
    'female': '#ba96e2',         # purple /russia
    'non-binary': '#4CAF50',     # Green
    'prefer not to say': '#98BACF'  # Gray
}

COLLABORATION_COLORS = {
    'Alone': '#98BACF',                              # Gray
    'With someone I knew': '#ba96e2',                # Purple
    'With someone I didn\'t know': '#1DB3B9',        # Teal
    'Played with Someone I Knew': '#ba96e2',         # Purple
    'Played with Someone I Didn\'t Know': '#1DB3B9'  # Teal
}

# Simplified Sankey colors (6 max)
SANKEY_SIMPLIFIED = {
    'Played with Someone I Knew': '#ba96e2',
    'Played with Someone I Didn\'t Know': '#1DB3B9',
    'Fun': '#4CAF50', 'Understand': '#FFC107', 'Enjoy': '#0068c9',
    'Fun: Positive': '#4CAF50', 'Fun: Negative': '#ffc932',
    'Understand: Positive': '#FFC107', 'Understand: Negative': '#ffc932',  
    'Enjoy: Positive': '#0068c9', 'Enjoy: Negative': '#ffc932'
}

# Central color mapping for Sankey nodes
sankey_color_map = {
    'Netherlands': '#6A60A9', 'Canada': '#1DB3B9', 'Belgium': '#F57274', 'Italy': '#FFC107',
    'Mexico': '#4CAF50', 'Russia': '#ba96e2', 'Germany': '#009688', 'China': '#F44336',
    'United Kingdom': '#673AB7', 'United States': '#03A9F4', 'Switzerland': '#E91E63',
    'Austria': '#F7B2AD', 'France': '#3F51B5', 'Sweden': '#8BC34A', 'European Union': '#FFB300',
    'Japan': '#FF9800', 'Hong Kong': '#009688', 'South Korea': '#3F51B5', 'Spain': '#00BCD4'
}


def hex_to_rgba(hex_color, alpha=0.3):
    """Convert hex color to rgba with transparency."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'



# =====================================================
# DATA LOADING AND CACHING
# =====================================================

@st.cache_data
def load_and_clean_data(uploaded_file=None):
    """Load and clean the MayRA survey data with caching"""
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Try to load from local file
        try:
            df = pd.read_csv('MayRASurvey_May232025_14.28.csv')
                # 🛠 Drop completely empty rows (extra blank lines in CSV)
            df = df.dropna(how='all')
        except FileNotFoundError:
            # Create sample data for demonstration
            st.warning("Data file not found. Using sample data for demonstration.")
            return create_sample_data()


    # Define value mappings
    likert_mapping = {1: "YES!", 2: "yes", 3: "no", 4: "NO!"}
    binary_mapping = {1: "yes", 2: "no"}
    grade_mapping = {1: "K-4th", 2: "5th", 3: "6th", 4: "7th", 5: "8th", 6: "9-12th", 7: "adult"}
    gender_mapping = {1: "male", 2: "female", 3: "non-binary", 4: "prefer not to say"}
    screen_mapping = {1: "left", 2: "right"}
    ethnicity_mapping = {1: "American Indian/Alaska Native", 2: "Asian", 3: "Black/African American", 
                         4: "Middle Eastern American", 5: "Native Hawaiian/Pacific Islander", 
                         6: "White", 7: "Other"}
    
    # Apply mappings
    df['Q3_have_fun_label'] = df['Q3_have_fun'].map(likert_mapping)
    df['Q4_good_job_label'] = df['Q4_good_job'].map(likert_mapping)
    df['Q5_do_cs_label'] = df['Q5_do_cs'].map(likert_mapping)
    df['Q6_enjoyed_fig_label'] = df['Q6_enjoyed_fig'].map(likert_mapping)
    df['Q8-Yes-diff-animal_label'] = df['Q8-Yes-diff-animal'].map(binary_mapping)
    df['Q9_knew_solve_label'] = df['Q9_knew_solve'].map(likert_mapping)
    df['Q10_fun_more_label'] = df['Q10_fun_more'].map(likert_mapping)
    df['Q11_understand_more_label'] = df['Q11_understand_more'].map(likert_mapping)
    df['Q12_enjoy_more_label'] = df['Q12_enjoy_more'].map(likert_mapping)
    
    df['Q7_difff_animals_label'] = df['Q7_difff_animals'].map(binary_mapping)
    df['Q19_read_ai_label'] = df['Q19_read_ai'].map(binary_mapping)
    
    df['Q13_grade_label'] = df['Q13_grade'].map(grade_mapping)
    df['Q14_gender_label'] = df['Q14_gender'].map(gender_mapping)
    df['Q1_screen_label'] = df['Q1_screen'].map(screen_mapping)
    # More robust ethnicity mapping
    df['Q15_ethnicity_label'] = df['Q15_ethnicity'].map(ethnicity_mapping).fillna('Unknown')
    # ADD NEW AI PERCEPTION MAPPINGS HERE:
    df['Q20_ai_interaction_label'] = df['Q20_ai_interaction'].map(likert_mapping)
    df['Q22_ai_understand_label'] = df['Q22_ai_understand'].map(likert_mapping)
    df['Q23_ai_exp_label'] = df['Q23_ai_exp'].map(likert_mapping)
    # Add binary flags for each ethnicity category
    df['Is_AIAN'] = df['Q15_ethnicity'].astype(str).str.contains('1').astype(int)
    df['Is_Asian'] = df['Q15_ethnicity'].astype(str).str.contains('2').astype(int)
    df['Is_Black'] = df['Q15_ethnicity'].astype(str).str.contains('3').astype(int)
    df['Is_MiddleEastern'] = df['Q15_ethnicity'].astype(str).str.contains('4').astype(int)
    df['Is_PacificIslander'] = df['Q15_ethnicity'].astype(str).str.contains('5').astype(int)
    df['Is_White'] = df['Q15_ethnicity'].astype(str).str.contains('6').astype(int)
    df['Is_Other'] = df['Q15_ethnicity'].astype(str).str.contains('7').astype(int)


    
    # Create composite scores
    df['engagement_score'] = (df['Q3_have_fun'] + df['Q6_enjoyed_fig']) / 2
    collaboration_cols = ['Q10_fun_more', 'Q11_understand_more', 'Q12_enjoy_more']
    df['collaboration_score'] = df[collaboration_cols].mean(axis=1)
    
    return df

# =====================================================
# Collaboration Analysis Function
# =====================================================
import pandas as pd

def prepare_collaboration_groups(df, include_questions=None):
    """
    Prepares data for Kruskal-Wallis analysis.
    Filters to participants with a single selection in Q2 (1, 2, or 3).
    Includes specified questions.
    """
    if include_questions is None:
        include_questions = ['Q10', 'Q11', 'Q12']  # Default to collab-specific questions

    if 'Q2_play_with' not in df.columns:
        st.warning("Q2_play_with column not found.")
        return None

    group_data = []
    for _, row in df.iterrows():
        q2_val = str(row.get('Q2_play_with', '')).strip()
        if pd.isna(q2_val) or q2_val == '':
            continue
        selections = [s.strip() for s in q2_val.split(',')]
        if len(selections) == 1:
            group = selections[0]
            if group == '1':
                group_label = 'Alone'
            elif group == '2':
                group_label = 'With Known'
            elif group == '3':
                group_label = 'With Unknown'
            else:
                continue
            entry = {'Group': group_label}
            for q in include_questions:
                entry[q] = row.get(q, None)
            group_data.append(entry)

    group_df = pd.DataFrame(group_data)
    return group_df



def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'Q3_have_fun': np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
        'Q6_enjoyed_fig': np.random.choice([1, 2, 3, 4], n_samples, p=[0.45, 0.3, 0.2, 0.05]),
        'Q4_good_job': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.25, 0.05]),
        'Q9_knew_solve': np.random.choice([1, 2, 3, 4], n_samples, p=[0.25, 0.45, 0.25, 0.05]),
        'Q7_difff_animals': np.random.choice([1, 2], n_samples, p=[0.7, 0.3]),
        'Q13_grade': np.random.choice([1, 2, 3, 4, 5, 6, 7], n_samples),
        'Q14_gender': np.random.choice([1, 2, 3, 4], n_samples, p=[0.45, 0.45, 0.05, 0.05]),
        'Q15_ethnicity': np.random.choice([1, 2, 3, 4, 5, 6, 7], n_samples),
    }
    return pd.DataFrame(data)

# =====================================================
# VISUALIZATION FUNCTIONS
# =====================================================

def create_ordered_likert_chart(df, column, label_column, title):
    """Create interactive Likert scale visualization with proper order (YES!, yes, no, NO!)"""
    
    counts = df[label_column].value_counts()
    
    # Define the correct order and colors
    likert_order = ['YES!', 'yes', 'no', 'NO!']
    color_map = {'YES!': '#0068c9', 'yes': '#82c7fd', 'no': '#feebb0', 'NO!': '#ffc932'}
    
    # Reorder data according to desired order
    ordered_data = []
    ordered_colors = []
    for response in likert_order:
        if response in counts.index:
            ordered_data.append((response, counts[response]))
            ordered_colors.append(color_map[response])
    
    if not ordered_data:  # Fallback if no data
        return go.Figure()
    
    responses, values = zip(*ordered_data)
    
    fig = go.Figure(data=[
        go.Bar(x=values, y=responses, orientation='h',
               marker_color=ordered_colors,
               text=[f'{val} ({val/counts.sum()*100:.1f}%)' for val in values],
               textposition='inside')
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Number of Responses",
        yaxis_title="Response",
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'array', 'categoryarray': list(reversed(likert_order))}  # Reverse for proper display
    )
    
    return fig

# =====================================================
# ALTERNATIVE VERSION: Percentage-based Diverging Chart
# =====================================================

def create_diverging_engagement_percentage_chart(df):
    """Create diverging bar chart showing percentage split from center"""
    
    questions = [
        ('Q3_have_fun_label', 'Q3: I was having fun'),
        ('Q6_enjoyed_fig_label', 'Q6: I enjoyed figuring things out')
    ]
    
    available_questions = [(col, label) for col, label in questions if col in df.columns]
    
    if not available_questions:
        return go.Figure()
    
    fig = go.Figure()
    
    colors = {
        'YES!': '#0068c9', 'yes': '#83c8fe',
        'no': '#FFEAB0', 'NO!': '#ffc932'
    }
    
    y_positions = list(range(len(available_questions)))
    question_labels = [label for _, label in available_questions]
    
    for response_type in ['YES!', 'yes', 'no', 'NO!']:
        x_values = []
        y_values = []
        text_values = []
        
        for i, (col, label) in enumerate(available_questions):
            count = (df[col] == response_type).sum()
            total = df[col].notna().sum()
            percentage = (count / total * 100) if total > 0 else 0
            
            # Make negative responses negative percentages
            if response_type in ['no', 'NO!']:
                x_value = -percentage
                text_value = f'{percentage:.1f}%'
            else:
                x_value = percentage
                text_value = f'{percentage:.1f}%'
            
            x_values.append(x_value)
            y_values.append(i)
            text_values.append(text_value)
        
        fig.add_trace(go.Bar(
            name=response_type,
            x=x_values,
            y=y_values,
            orientation='h',
            marker_color=colors[response_type],
            text=text_values,
            textposition='outside',
            textfont=dict(size=10)
        ))
    
    # Add vertical line at x=0
    fig.add_vline(x=0, line_width=2, line_color="black", opacity=0.8)
    
    fig.update_layout(
        title='Engagement Balance: Positive vs Negative (%)',
        xaxis=dict(
            title="← Negative % | Positive % →",
            range=[-100, 100],  # Fixed range for percentage
            tickformat='.0f',
            ticksuffix='%'
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=y_positions,
            ticktext=question_labels
        ),
        barmode='relative',
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig

def create_engagement_grouped_chart(df):
    """Create grouped bar chart comparing Q3 and Q6 responses side by side"""
    
    # Get value counts for both questions
    q3_counts = df['Q3_have_fun_label'].value_counts() if 'Q3_have_fun_label' in df.columns else pd.Series()
    q6_counts = df['Q6_enjoyed_fig_label'].value_counts() if 'Q6_enjoyed_fig_label' in df.columns else pd.Series()
    
    if q3_counts.empty or q6_counts.empty:
        return go.Figure()
    
    # Define the correct order and colors
    likert_order = ['YES!', 'yes', 'no', 'NO!']
    color_map = {'Q3': '#6446a0', 'Q6': '#ffc00a'}  # Different colors for each question
    
    # Prepare data for both questions
    q3_values = [q3_counts.get(cat, 0) for cat in likert_order]
    q6_values = [q6_counts.get(cat, 0) for cat in likert_order]
    
    fig = go.Figure()
    
    # Add Q3 bars
    fig.add_trace(go.Bar(
        name='Q3: Having Fun',
        x=likert_order,
        y=q3_values,
        marker_color=color_map['Q3'],
        text=[f'{val}' for val in q3_values],
        textposition='outside',
        offsetgroup=1
    ))
    
    # Add Q6 bars
    fig.add_trace(go.Bar(
        name='Q6: Enjoyed Challenges',
        x=likert_order,
        y=q6_values,
        marker_color=color_map['Q6'],
        text=[f'{val}' for val in q6_values],
        textposition='outside',
        offsetgroup=2
    ))
    
    fig.update_layout(
        title='Engagement Comparison: Having Fun vs Enjoying Challenges',
        xaxis_title='Response Categories',
        yaxis_title='Number of Responses',
        barmode='group',
        height=500,
        legend_title='Questions',
        xaxis={'categoryorder': 'array', 'categoryarray': likert_order}
    )
    
    return fig

def create_stacked_bar_chart(df, questions, titles, main_title):
    """Create stacked bar chart for multiple questions"""
    
    # Prepare data
    plot_data = []
    for q, title in zip(questions, titles):
        counts = df[q].value_counts()
        for response, count in counts.items():
            plot_data.append({
                'Question': title,
                'Response': response,
                'Count': count
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create stacked bar chart
    fig = px.bar(plot_df, x='Question', y='Count', color='Response',
                 title=main_title,
                 color_discrete_map={'YES!': '#2E8B57', 'yes': '#90EE90', 
                                   'no': '#FFB6C1', 'NO!': '#DC143C'})
    
    fig.update_layout(height=500, xaxis_tickangle=-45)
    
    return fig

### Collab charts data prep
def create_multiselect_collaboration_chart(df, column, title):
    """Create collaboration chart handling multi-select responses for Q2"""
    
    collaboration_mapping = {
        '1': "Alone",
        '2': "With someone I knew", 
        '3': "With someone I didn't know"
    }
    
    # Count each collaboration type
    collab_counts = {}
    
    for value in df[column].dropna():
        if pd.isna(value) or value == '':
            continue
            
        # Split by comma and count each selection
        selections = str(value).split(',')
        for selection in selections:
            selection = selection.strip()
            if selection in collaboration_mapping:
                label = collaboration_mapping[selection]
                collab_counts[label] = collab_counts.get(label, 0) + 1
    
    if not collab_counts:
        return go.Figure()
    
    # Convert to series for plotting
    counts_series = pd.Series(collab_counts).sort_values(ascending=True)
    
    colors = [COLLABORATION_COLORS.get(label, '#78909C') for label in counts_series.index]

    
    fig = go.Figure(data=[
        go.Bar(
            x=counts_series.values,
            y=counts_series.index,
            orientation='h',
            marker_color=colors,
            text=[f'{val} ({val/counts_series.sum()*100:.1f}%)' for val in counts_series.values],
            textposition='outside',
            textfont=dict(size=10)
        )
    ])
    
    fig.update_layout(
        title=f"{title}<br><sub>Note: Respondents could select multiple options</sub>",
        xaxis_title="Number of Responses",
        yaxis_title="Collaboration Type",
        height=400,
        showlegend=False,
        margin=dict(l=200, r=50, t=80, b=50)
    )
    
    return fig
### Collab Sankey
def create_collaboration_sankey_proportional(df):
    """Create proportional Sankey with improved coloring and readability"""

    # Mapping for colors (customize if needed)
    sankey_color_map = {
        'Played with Someone I Knew': '#ba96e2',
        'Played with Someone I Didn\'t Know': '#1DB3B9',
        'Fun': '#F57274', 'Understand': '#FFC107', 'Enjoy': '#4CAF50'
    }

    def hex_to_rgba(hex_color, alpha=0.3):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

    positive_responses = ['YES!', 'yes']
    negative_responses = ['no', 'NO!']

    sankey_data = []
    for _, row in df.iterrows():
        q2_val = str(row.get('Q2_play_with', ''))
        if pd.isna(q2_val) or q2_val == '' or q2_val == 'nan':
            continue
        selections = [s.strip() for s in q2_val.split(',')]
        collab_types = []
        if '2' in selections:
            collab_types.append('Played with Someone I Knew')
        if '3' in selections:
            collab_types.append('Played with Someone I Didn\'t Know')
        q10_resp = row.get('Q10_fun_more_label')
        q11_resp = row.get('Q11_understand_more_label')
        q12_resp = row.get('Q12_enjoy_more_label')
        for collab_type in collab_types:
            sankey_data.append({
                'collaboration': collab_type,
                'q10_response': q10_resp,
                'q11_response': q11_resp,
                'q12_response': q12_resp
            })

    if not sankey_data:
        return go.Figure()

    sankey_df = pd.DataFrame(sankey_data)
    result_data = []
    for collab_type in ['Played with Someone I Knew', 'Played with Someone I Didn\'t Know']:
        collab_data = sankey_df[sankey_df['collaboration'] == collab_type]
        if collab_data.empty:
            continue
        total_for_group = len(collab_data)
        outcomes = [
            ('Fun: Positive', collab_data['q10_response'].isin(positive_responses).sum()),
            ('Fun: Negative', collab_data['q10_response'].isin(negative_responses).sum()),
            ('Understand: Positive', collab_data['q11_response'].isin(positive_responses).sum()),
            ('Understand: Negative', collab_data['q11_response'].isin(negative_responses).sum()),
            ('Enjoy: Positive', collab_data['q12_response'].isin(positive_responses).sum()),
            ('Enjoy: Negative', collab_data['q12_response'].isin(negative_responses).sum()),
        ]
        for outcome_name, count in outcomes:
            if count > 0:
                percentage = (count / total_for_group) * 100
                result_data.append({
                    'source': collab_type,
                    'target': outcome_name,
                    'value': percentage,
                    'count': count,
                    'total': total_for_group
                })

    if not result_data:
        return go.Figure()

    all_labels = ['Played with Someone I Knew', 'Played with Someone I Didn\'t Know',
                  'Fun: Positive', 'Fun: Negative',
                  'Understand: Positive', 'Understand: Negative',
                  'Enjoy: Positive', 'Enjoy: Negative']

    # Node colors
    node_colors = [sankey_color_map.get(label.split(':')[0].strip(), '#cccccc') for label in all_labels]

    # Link colors (lighter version of source node color)
    link_colors = [hex_to_rgba(sankey_color_map.get(item['source'].split(':')[0].strip(), '#cccccc')) for item in result_data]

    # Build arrays
    source_indices = []
    target_indices = []
    values = []
    hover_texts = []
    for item in result_data:
        source_idx = all_labels.index(item['source'])
        target_idx = all_labels.index(item['target'])
        source_indices.append(source_idx)
        target_indices.append(target_idx)
        values.append(item['value'])
        hover_texts.append(f"Count: {item['count']}/{item['total']}<br>Percentage: {item['value']:.1f}%")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors,
            customdata=hover_texts,
            hovertemplate='%{source.label} → %{target.label}<br>%{customdata}<extra></extra>'
        )
    )])

    fig.update_layout(
        title="Collaboration → Outcomes (% within each group)",
        font=dict(size=12),
        height=700,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig
    """Create proportional Sankey with gradient colors and improved readability"""
    
    # Just use the original function but modify the final figure creation
    positive_responses = ['YES!', 'yes']
    negative_responses = ['no', 'NO!']
    
    # Prepare the data
    sankey_data = []
    
    for _, row in df.iterrows():
        q2_val = str(row.get('Q2_play_with', ''))
        if pd.isna(q2_val) or q2_val == '' or q2_val == 'nan':
            continue
            
        selections = [s.strip() for s in q2_val.split(',')]
        
        collab_types = []
        if '2' in selections:
            collab_types.append('Played with Someone I Knew')
        if '3' in selections:
            collab_types.append('Played with Someone I Didn\'t Know')
            
        q10_resp = row.get('Q10_fun_more_label')
        q11_resp = row.get('Q11_understand_more_label') 
        q12_resp = row.get('Q12_enjoy_more_label')
        
        for collab_type in collab_types:
            sankey_data.append({
                'collaboration': collab_type,
                'q10_response': q10_resp,
                'q11_response': q11_resp,
                'q12_response': q12_resp
            })
    
    if not sankey_data:
        return go.Figure()
    
    sankey_df = pd.DataFrame(sankey_data)
    
    # Calculate proportions by collaboration type
    result_data = []
    
    for collab_type in ['Played with Someone I Knew', 'Played with Someone I Didn\'t Know']:
        collab_data = sankey_df[sankey_df['collaboration'] == collab_type]
        if collab_data.empty:
            continue
            
        total_for_group = len(collab_data)
        
        # Calculate for each outcome
        outcomes = [
            ('Fun: Positive', collab_data['q10_response'].isin(positive_responses).sum()),
            ('Fun: Negative', collab_data['q10_response'].isin(negative_responses).sum()),
            ('Understand: Positive', collab_data['q11_response'].isin(positive_responses).sum()),
            ('Understand: Negative', collab_data['q11_response'].isin(negative_responses).sum()),
            ('Enjoy: Positive', collab_data['q12_response'].isin(positive_responses).sum()),
            ('Enjoy: Negative', collab_data['q12_response'].isin(negative_responses).sum()),
        ]
        
        for outcome_name, count in outcomes:
            if count > 0:
                percentage = (count / total_for_group) * 100
                result_data.append({
                    'source': collab_type,
                    'target': outcome_name,
                    'value': percentage,
                    'count': count,
                    'total': total_for_group
                })
    
    if not result_data:
        return go.Figure()
    
    # Create labels and indices
    
    all_labels = ['Played with Someone I Knew', 'Played with Someone I Didn\'t Know',
              'Fun: Positive', 'Fun: Negative', 'Understand: Positive', 'Understand: Negative', 
              'Enjoy: Positive', 'Enjoy: Negative']

    node_colors = [sankey_color_map.get(label.split(':')[0].strip(), '#cccccc') for label in all_labels]
    link_colors = [hex_to_rgba(sankey_color_map.get(item['source'].split(':')[0].strip(), '#cccccc')) for item in result_data]


    # Build the final arrays
    source_indices = []
    target_indices = []
    values = []
    hover_texts = []
    
    for item in result_data:
        source_idx = source_labels.index(item['source'])
        target_idx = len(source_labels) + target_labels.index(item['target'])
        
        source_indices.append(source_idx)
        target_indices.append(target_idx)
        values.append(item['value'])
        hover_texts.append(f"Count: {item['count']}/{item['total']}<br>Percentage: {item['value']:.1f}%")
    
    # Colors matching your theme
    source_colors = ["#667eea", "#764ba2"]
    target_colors = [
        "#2E8B57", "#DC143C", "#2E8B57", "#DC143C", 
        "#2E8B57", "#DC143C"
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=source_colors + target_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            hovertemplate='%{source.label} → %{target.label}<br>%{customdata}<extra></extra>',
            customdata=hover_texts,
            color="rgba(102, 126, 234, 0.3)"
        )
    )])
    
    # IMPROVED FONT SETTINGS
    fig.update_layout(
        title={
            'text': "Collaboration → Outcomes (% within each group)",
            'font': {'size': 18, 'family': 'Arial Black', 'color': '#2c3e50'}
        },
        font={'size': 14, 'family': 'Arial', 'color': '#2c3e50'},  # Larger, darker font
        height=700,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig



## Collab Horizontal Bar Chart
def create_collaboration_stacked_bars(df):
    """Create horizontal stacked bar chart for collaboration outcomes"""
    
    # Prepare data similar to Sankey but for stacked bars
    collab_data = {'Played with Someone I Knew': [], 'Played with Someone I Didn\'t Know': []}    
    
    for _, row in df.iterrows():
        q2_val = str(row.get('Q2_play_with', ''))
        if pd.isna(q2_val) or q2_val == '' or q2_val == 'nan':
            continue
            
        selections = [s.strip() for s in q2_val.split(',')]
        
        row_data = {
            'q10': row.get('Q10_fun_more_label'),
            'q11': row.get('Q11_understand_more_label'),
            'q12': row.get('Q12_enjoy_more_label'),
            'q7': row.get('Q7_difff_animals_label')
        }
        
        if '2' in selections:
            collab_data['Played with Someone I Knew'].append(row_data)
        if '3' in selections:
            collab_data['Played with Someone I Didn\'t Know'].append(row_data)
    
    # Create subplot with 3 charts (one for each question)
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['Q10: Fun with Others', 'Q11: Understanding with Others', 
                       'Q12: Enjoyment with Others'],
        vertical_spacing=0.08
    )
    
    questions = ['q10', 'q11', 'q12']
    question_titles = ['Fun', 'Understanding', 'Enjoyment']
    
    # Colors matching your theme
    colors = {'YES!': '#0068c9', 'yes': '#83c8fe', 'NO!': '#ffc932', 'no': '#FFEAB0'}  # For Q7
    
    for row_idx, (q, title) in enumerate(zip(questions, question_titles)):
        for collab_type in ['Played with Someone I Knew', 'Played with Someone I Didn\'t Know']:
            responses = [item[q] for item in collab_data[collab_type] if item[q] is not None]
            
            if not responses:
                continue
                
            # Count responses
            response_counts = pd.Series(responses).value_counts()
            total = len(responses)
            
            # Convert to percentages
            percentages = (response_counts / total * 100).round(1)
            
            # Create stacked bar data
            x_data = []
            colors_data = []
            text_data = []
            
            cumulative = 0
            for response in ['YES!', 'yes', 'no', 'NO!']:
                if response in percentages:
                    x_data.append(percentages[response])
                    colors_data.append(colors.get(response, '#888888'))
                    text_data.append(f'{percentages[response]:.1f}%')
                    cumulative += percentages[response]
                else:
                    x_data.append(0)
                    colors_data.append('#888888')
                    text_data.append('')
            
            # Add trace for each response type
            for i, response in enumerate(['YES!', 'yes', 'no', 'NO!']):
                if response in response_counts:
                    fig.add_trace(
                        go.Bar(
                            name=response,  # Only show legend for first row
                            x=[percentages[response]],
                            y=[collab_type],
                            orientation='h',
                            marker_color=colors[response],
                            text=f'{percentages[response]:.1f}%',
                            textposition='inside',
                            showlegend=(row_idx == 0 and collab_type == 'Played with Someone I Knew'),  # Only show legend for first subplot
                            # offsetgroup=collab_type,
                            base=sum(percentages[resp] for resp in ['YES!', 'yes', 'no', 'NO!'][:i] if resp in percentages)
                        ),
                        row=row_idx + 1, col=1
                    )
    
    fig.update_layout(
        title="Collaboration Type vs Experience",
        barmode='stack',
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update x-axes to show percentages
    for i in range(1, 4):
        fig.update_xaxes(title_text="", range=[0, 100], row=i, col=1)
    
    return fig
    

def create_vertical_bar_chart(df, column, label_column, title):
    """Create vertical bar chart with unified colors"""
    
    counts = df[label_column].value_counts()
    
    if counts.empty:
        return go.Figure()
    
    # Sort by logical order for grades
    grade_order = ['K-4th', '5th', '6th', '7th', '8th', '9-12th', 'adult']
    
    # Reorder if it's grade data
    if 'grade' in title.lower():
        available_grades = [grade for grade in grade_order if grade in counts.index]
        if available_grades:
            counts = counts.reindex(available_grades)
    
    # REPLACE this line:
    # OLD: colors = px.colors.qualitative.Set3[:len(counts)]
    # NEW: Use sequential colors for grades (light to dark blue)
    if 'grade' in title.lower():
        # Sequential blue colors for grades (K-4th = lightest, adult = darkest)
        grade_colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1976D2']
        colors = [grade_colors[i] for i in range(len(counts))]
    else:
        # Use selective palette for other vertical charts
        palette = ['#1DB3B9', '#FFC107', '#4CAF50', '#6A60A9', '#0068c9', '#78909C']
        colors = [palette[i % len(palette)] for i in range(len(counts))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=counts.index,  # Grade labels on x-axis
            y=counts.values, 
            marker_color=colors,
            text=[f'{val}<br>({val/counts.sum()*100:.1f}%)' for val in counts.values],
            textposition='outside',
            textfont=dict(size=10)
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Grade Level",
        yaxis_title="Number of Responses",
        height=400,
        showlegend=False,
        xaxis={'categoryorder': 'array', 'categoryarray': grade_order}
    )
    
    return fig  
    
    counts = df[label_column].value_counts()
    
    if counts.empty:
        return go.Figure()
    
    # Sort by logical order for grades
    grade_order = ['K-4th', '5th', '6th', '7th', '8th', '9-12th', 'adult']
    
    # Reorder if it's grade data
    if 'grade' in title.lower():
        available_grades = [grade for grade in grade_order if grade in counts.index]
        if available_grades:
            counts = counts.reindex(available_grades)
    
    # Use a nice color palette
    colors = px.colors.qualitative.Set3[:len(counts)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=counts.index,  # Grade labels on x-axis
            y=counts.values, 
            marker_color=colors,
            text=[f'{val}<br>({val/counts.sum()*100:.1f}%)' for val in counts.values],
            textposition='outside',
            textfont=dict(size=10)
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Grade Level",
        yaxis_title="Number of Responses",
        height=400,
        showlegend=False,
        xaxis={'categoryorder': 'array', 'categoryarray': grade_order}
    )
    
    return fig

def create_ethnicity_bar_chart(df, label_column, title):
    """Create horizontal bar chart for ethnicity with full labels"""
    
    counts = df[label_column].value_counts()
    
    if counts.empty:
        return go.Figure()
    
    # Use a diverse color palette
    colors = px.colors.qualitative.Set3[:len(counts)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=counts.values,
            y=counts.index,
            orientation='h',
            marker_color=colors,
            text=[f'{val} ({val/counts.sum()*100:.1f}%)' for val in counts.values],
            textposition='outside',
            textfont=dict(size=10)
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Number of Responses",
        yaxis_title="Ethnicity",
        height=500,
        showlegend=False,
        margin=dict(l=200, r=50, t=50, b=50),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_ethnicity_bar_chart_numeric(df, column, title):
    """Create ethnicity bar chart from numeric values with proper mapping"""
    
    counts = df[column].value_counts()
    
    if counts.empty:
        return go.Figure()
    
    # Ethnicity mapping
    ethnicity_mapping = {
        1: "American Indian/Alaska Native",
        2: "Asian", 
        3: "Black/African American",
        4: "Middle Eastern American",
        5: "Native Hawaiian/Pacific Islander",
        6: "White",
        7: "Other"
    }
    
    # Map numeric values to labels
    mapped_data = {}
    for numeric_val, count in counts.items():
        label = ethnicity_mapping.get(numeric_val, f"Category {numeric_val}")
        mapped_data[label] = count
    
    # Create series for plotting
    mapped_counts = pd.Series(mapped_data)
    
    # Use a diverse color palette
    colors = px.colors.qualitative.Set3[:len(mapped_counts)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=mapped_counts.values,
            y=mapped_counts.index,
            orientation='h',
            marker_color=colors,
            text=[f'{val} ({val/mapped_counts.sum()*100:.1f}%)' for val in mapped_counts.values],
            textposition='outside',
            textfont=dict(size=10)
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Number of Responses",
        yaxis_title="Ethnicity",
        height=500,
        showlegend=False,
        margin=dict(l=200, r=50, t=50, b=50),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_donut_chart(df, column, label_column, title, chart_type="likert"):
    """Create donut chart with unified color system"""
    
    counts = df[label_column].value_counts()
    
    if counts.empty:
        return go.Figure()
    
    # Get colors based on chart type
    if chart_type == "likert":
        colors = [LIKERT_COLORS.get(label, '#cccccc') for label in counts.index]
    elif chart_type == "gender":
        colors = [DEMOGRAPHIC_COLORS.get(label, '#cccccc') for label in counts.index]
    else:
        # For other demographics, cycle through selective palette
        palette = ['#1DB3B9', '#FFC107', '#4CAF50', '#ba96e2', '#0068c9', '#78909C']
        colors = [palette[i % len(palette)] for i in range(len(counts))]
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=.6,
        textinfo="label+percent",
        textposition="outside",
        marker=dict(colors=colors)
    )])
    
    fig.update_layout(
        title=title,
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5)
    )
    
    return fig
    """Create donut chart with custom colors for positive (blue) and negative (yellow) responses"""
    
    counts = df[label_column].value_counts()
    
    # Final custom color mapping
    color_map = {'YES!': '#0068c9', 'yes': '#83c8fe', 'NO!': '#ffc932', 'no': '#FFEAB0'}
    colors = [color_map.get(label, '#cccccc') for label in counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=.6,
        textinfo="label+percent",
        textposition="outside",
        marker=dict(colors=colors)
    )])
    
    fig.update_layout(
        title=title,
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5)
    )
    
    return fig


def create_correlation_heatmap(df, columns, labels, title):
    """Create correlation heatmap"""
    
    corr_data = df[columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=labels,
        y=labels,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(corr_data.values, 3),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=500
    )
    
    return fig

def create_horizontal_bar_chart(df, column, label_column, title):
    """Create horizontal bar chart for categories with long labels (like grade levels)"""
    
    counts = df[label_column].value_counts()
    
    if counts.empty:
        return go.Figure()
    
    # Sort by a logical order for grades if possible
    grade_order = ['K-4th', '5th', '6th', '7th', '8th', '9-12th', 'Adult']
    
    # Reorder if it's grade data
    if 'grade' in title.lower():
        available_grades = [grade for grade in grade_order if grade in counts.index]
        if available_grades:
            counts = counts.reindex(available_grades)
    
    # Use a nice color palette
    colors = px.colors.qualitative.Set3[:len(counts)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=counts.values, 
            y=counts.index, 
            orientation='h',
            marker_color=colors,
            text=[f'{val} ({val/counts.sum()*100:.1f}%)' for val in counts.values],
            textposition='inside',
            textfont=dict(size=10)
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Number of Responses",
        yaxis_title="Categories",
        height=400,
        showlegend=False,
        margin=dict(l=100, r=50, t=50, b=50)  # More left margin for labels
    )
    
    return fig

# Updated create_animal_strategy_sankey function  
def create_animal_strategy_sankey(df):
    """Create 3-node Sankey: Animal Usage → Garden Better → Understanding (Positive/Negative) with improved coloring"""

    sankey_color_map = {
        'All Participants': '#ba96e2',
        'Used Different: Yes': '#1DB3B9',
        'Used Different: No': '#F57274',
        'Garden Better: Yes': '#FFC107',
        'Garden Better: No': '#4CAF50',
        'Understanding: Positive': '#ba96e2',
        'Understanding: Negative': '#FF9800'
    }

    def hex_to_rgba(hex_color, alpha=0.3):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

    sankey_data = []
    for _, row in df.iterrows():
        q7_resp = row.get('Q7_difff_animals_label')
        q8_resp = row.get('Q8-Yes-diff-animal_label')
        q9_resp = row.get('Q9_knew_solve_label')
        if pd.isna(q7_resp) or pd.isna(q9_resp):
            continue
        sankey_data.append({
            'q7_response': q7_resp,
            'q8_response': q8_resp if q7_resp == 'yes' else None,
            'q9_response': q9_resp
        })

    if not sankey_data:
        return go.Figure()

    sankey_df = pd.DataFrame(sankey_data)
    all_labels = [
        'All Participants', 'Used Different: Yes', 'Used Different: No',
        'Garden Better: Yes', 'Garden Better: No',
        'Understanding: Positive', 'Understanding: Negative'
    ]

    source_indices, target_indices, values = [], [], []
    positive_responses = ['YES!', 'yes']
    negative_responses = ['no', 'NO!']

    q7_yes_count = len(sankey_df[sankey_df['q7_response'] == 'yes'])
    q7_no_count = len(sankey_df[sankey_df['q7_response'] == 'no'])
    if q7_yes_count > 0:
        source_indices.append(0)
        target_indices.append(1)
        values.append(q7_yes_count)
    if q7_no_count > 0:
        source_indices.append(0)
        target_indices.append(2)
        values.append(q7_no_count)

    q7_yes_data = sankey_df[sankey_df['q7_response'] == 'yes']
    q8_yes_count = len(q7_yes_data[q7_yes_data['q8_response'] == 'yes'])
    q8_no_count = len(q7_yes_data[q7_yes_data['q8_response'] == 'no'])
    if q8_yes_count > 0:
        source_indices.append(1)
        target_indices.append(3)
        values.append(q8_yes_count)
    if q8_no_count > 0:
        source_indices.append(1)
        target_indices.append(4)
        values.append(q8_no_count)

    for q8_val, source_idx in [('yes', 3), ('no', 4)]:
        data = q7_yes_data[q7_yes_data['q8_response'] == q8_val]
        if not data.empty:
            pos_count = len(data[data['q9_response'].isin(positive_responses)])
            neg_count = len(data[data['q9_response'].isin(negative_responses)])
            if pos_count > 0:
                source_indices.append(source_idx)
                target_indices.append(5)
                values.append(pos_count)
            if neg_count > 0:
                source_indices.append(source_idx)
                target_indices.append(6)
                values.append(neg_count)

    q7_no_data = sankey_df[sankey_df['q7_response'] == 'no']
    pos_count = len(q7_no_data[q7_no_data['q9_response'].isin(positive_responses)])
    neg_count = len(q7_no_data[q7_no_data['q9_response'].isin(negative_responses)])
    if pos_count > 0:
        source_indices.append(2)
        target_indices.append(5)
        values.append(pos_count)
    if neg_count > 0:
        source_indices.append(2)
        target_indices.append(6)
        values.append(neg_count)

    node_colors = [sankey_color_map.get(label, '#cccccc') for label in all_labels]
    link_colors = [hex_to_rgba(sankey_color_map.get(all_labels[source], '#cccccc')) for source in source_indices]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors,
            hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Count: <b>%{value}</b><extra></extra>'
        )
    )])

    fig.update_layout(
        title={
            'text': "🦁 Animal Strategy → Garden Perception → Learning Outcome<br><sub>Conditional flow: Garden perception only applies to those who used different animals</sub>",
            'x': 0.5
        },
        font=dict(size=12),
        height=800,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig
    """Create 3-node Sankey: Animal Usage → Garden Better → Understanding (Positive/Negative)"""
    
def create_demographic_donut_chart(df, column, label_column, title, chart_type="general"):
    """Create donut chart with appropriate colors for different demographic types"""
    
    counts = df[label_column].value_counts()
    
    if counts.empty:
        return go.Figure()
    
    # Define color mappings for different chart types
    if chart_type == "gender":
        color_map = {
            'male': '#4472C4',
            'female': '#E15759', 
            'non-binary': '#70AD47',
            'prefer not to say': '#FFC000'
        }
    elif chart_type == "likert":
        color_map = {'YES!': '#0068c9', 'yes': '#83c8fe', 'NO!': '#ffc932', 'no': '#FFEAB0'}
    else:
        # Use a diverse color palette for other demographics
        colors = px.colors.qualitative.Set3[:len(counts)]
        color_map = {label: color for label, color in zip(counts.index, colors)}
    
    colors = [color_map.get(label, '#cccccc') for label in counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=.6,
        textinfo="label+percent",
        textposition="outside",
        marker=dict(colors=colors),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5),
        font=dict(size=12)
    )
    
    return fig
    

def create_multiselect_ethnicity_chart(df, column, title):
    """Create ethnicity chart handling multi-select responses (e.g., '1,2,3')"""
    
    ethnicity_mapping = {
        '1': "American Indian/Alaska Native",
        '2': "Asian", 
        '3': "Black/African American",
        '4': "Middle Eastern American",
        '5': "Native Hawaiian/Pacific Islander",
        '6': "White",
        '7': "Other"
    }
    
    # Count each ethnicity (people can select multiple)
    ethnicity_counts = {}
    
    for value in df[column].dropna():
        if pd.isna(value) or value == '':
            continue
            
        # Split by comma and count each selection
        ethnicity_counts = {
        'American Indian/Alaska Native': df['Is_AIAN'].sum(),
        'Asian': df['Is_Asian'].sum(),
        'Black/African American': df['Is_Black'].sum(),
        'Middle Eastern American': df['Is_MiddleEastern'].sum(),
        'Native Hawaiian/Pacific Islander': df['Is_PacificIslander'].sum(),
        'White': df['Is_White'].sum(),
        'Other': df['Is_Other'].sum()
        }

    
    if not ethnicity_counts:
        return go.Figure()
    
    # Convert to series for plotting
    counts_series = pd.Series(ethnicity_counts).sort_values(ascending=True)
    
    # Use a diverse color palette
    colors = px.colors.qualitative.Set3[:len(counts_series)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=counts_series.values,
            y=counts_series.index,
            orientation='h',
            marker_color=colors,
            text=[f'{val} ({val/counts_series.sum()*100:.1f}%)' for val in counts_series.values],
            textposition='outside',
            textfont=dict(size=10)
        )
    ])
    
    fig.update_layout(
        title=f"{title}<br><sub>Note: Respondents could select multiple ethnicities</sub>",
        xaxis_title="Number of Responses",
        yaxis_title="Ethnicity",
        height=500,
        showlegend=False,
        margin=dict(l=250, r=50, t=80, b=50)
    )
    
    return fig

def create_other_ethnicity_summary(df):
    # Filter for 'Other' responses and cleaned text
    other_rows = df[df['Q15_ethnicity'].astype(str).str.contains('7', na=False) & df['Q15_7_text'].notna()]
    
    # Count unique entries
    summary = other_rows['Q15_7_text'].value_counts().reset_index()
    summary.columns = ['Ethnicity Detail', 'Count']
    summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(1)
    
    return summary



import plotly.graph_objects as go
import plotly.express as px

def create_ethnicity_donut_chart(df, title="Ethnicity Distribution"):
    # Binary flag counts
    counts = {
        'American Indian/Alaska Native': df['Is_AIAN'].sum(),
        'Asian': df['Is_Asian'].sum(),
        'Black/African American': df['Is_Black'].sum(),
        'Middle Eastern American': df['Is_MiddleEastern'].sum(),
        'Native Hawaiian/Pacific Islander': df['Is_PacificIslander'].sum(),
        'White': df['Is_White'].sum(),
        'Other': df['Is_Other'].sum()
    }
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.6,
        textinfo="label+percent+value",
        textposition="outside",
        marker=dict(colors=px.colors.qualitative.Set3[:len(counts)])
    )])
    
    fig.update_layout(
        title=f"{title}<br><sub>Counts reflect unique respondents (binary flags)</sub>",
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5)
    )
    
    return fig
    """Create donut chart for multi-select ethnicity data"""
    
    ethnicity_mapping = {
        '1': "American Indian/Alaska Native",
        '2': "Asian", 
        '3': "Black/African American",
        '4': "Middle Eastern American",
        '5': "Native Hawaiian/Pacific Islander",
        '6': "White",
        '7': "Other"
    }
    
    # Count each ethnicity (people can select multiple)
    ethnicity_counts = {}
    
    for value in df[column].dropna():
        if pd.isna(value) or value == '':
            continue
            
        # Split by comma and count each selection
        selections = str(value).split(',')
        for selection in selections:
            selection = selection.strip()
            if selection in ethnicity_mapping:
                label = ethnicity_mapping[selection]
                ethnicity_counts[label] = ethnicity_counts.get(label, 0) + 1
    
    if not ethnicity_counts:
        return go.Figure()
    
    # Convert to series for plotting
    counts_series = pd.Series(ethnicity_counts)
    
    # NEW:
    palette = ['#1DB3B9', '#FFC107', '#4CAF50', '#6A60A9', '#0068c9', '#78909C', '#ffc932']
    colors = [palette[i % len(palette)] for i in range(len(counts_series))]

    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=list(counts_series.index), 
        values=list(counts_series.values),
        hole=.6,
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=10),  # Smaller text for potentially longer labels
        marker=dict(colors=colors)
    )])
    
    fig.update_layout(
        title=f"{title}<br><sub>Note: Multi-select responses</sub>",
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, font=dict(size=9))
    )
    
    return fig

# Other category of ethnicity breakdown
def create_other_ethnicity_breakdown(df):
    """Create breakdown table for 'Other' ethnicity responses"""
    
    # Get rows where Q15_ethnicity contains '7' and has text data
    other_ethnicities = []
    
    for _, row in df.iterrows():
        ethnicity_val = str(row.get('Q15_ethnicity', ''))
        if '7' in ethnicity_val:
            # Check both possible text columns
            other_text = row.get('Q15_7_text') or row.get('Q15_7_TEXT')
            if other_text and str(other_text).strip():
                other_ethnicities.append(str(other_text).strip())
    
    if not other_ethnicities:
        return None
    
    # Count occurrences and clean up similar entries
    other_counts = {}
    for ethnicity in other_ethnicities:
        # Basic cleaning: normalize case and spacing
        cleaned = ethnicity.strip()
        
        # Group similar entries (you can expand this logic)
        if cleaned.lower() in ['hispanic', 'hispanic ', 'latina', 'latino']:
            cleaned = 'Hispanic/Latino'
        elif cleaned.lower() in ['spanish']:
            cleaned = 'Spanish'
        elif cleaned.lower() in ['dominican', 'dominican republic', 'dominican republic ']:
            cleaned = 'Dominican'
        
        other_counts[cleaned] = other_counts.get(cleaned, 0) + 1
    
    # Convert to DataFrame and sort by count
    breakdown_df = pd.DataFrame([
        {'Ethnicity': ethnicity, 'Count': count}
        for ethnicity, count in other_counts.items()
    ]).sort_values('Count', ascending=False)
    
    # Add percentage
    total_other = breakdown_df['Count'].sum()
    breakdown_df['Percentage'] = (breakdown_df['Count'] / total_other * 100).round(1)
    breakdown_df['Display'] = breakdown_df.apply(
        lambda x: f"{x['Count']} ({x['Percentage']:.1f}%)", axis=1
    )
    
    return breakdown_df


### AI Perceptions
# ADD THESE FUNCTIONS TO YOUR CODE:

def create_ai_donut_chart(df, column, label_column, title, question_text):
    """Create donut chart for AI perception questions with enhanced styling"""
    
    # Filter for only participants who read about AI (Q19_read_ai = 1)
    ai_participants = df[df['Q19_read_ai'] == 1]
    
    if ai_participants.empty:
        return go.Figure().add_annotation(
            text="No AI participants found",
            x=0.5, y=0.5,
            showarrow=False,
            font_size=16
        )
    
    counts = ai_participants[label_column].value_counts()
    
    if counts.empty:
        return go.Figure()
    
    # Define colors matching your theme
    colors = [LIKERT_COLORS.get(response, '#cccccc') for response in counts.index]

    
    # Get colors for the data    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, 
        values=counts.values,
        hole=.6,
        textinfo="label+percent+value",
        textposition="outside",
        marker_colors=colors,
        textfont={'size': 12, 'family': 'Arial', 'color': '#2c3e50'},
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': f"{title}<br><sub>{question_text}</sub><br><sub style='color: #666;'>Based on {len(ai_participants)} participants who read about AI</sub>",
            'font': {'size': 16, 'family': 'Arial', 'color': '#2c3e50'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v", 
            yanchor="middle", 
            y=0.5,
            font={'size': 11, 'family': 'Arial', 'color': '#2c3e50'}
        ),
        font={'size': 12, 'family': 'Arial', 'color': '#2c3e50'},
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def create_ai_comparison_chart(df):
    """Create comparison chart for all three AI perception questions"""
    
    # Filter for AI participants
    ai_participants = df[df['Q19_read_ai'] == 1]
    
    if ai_participants.empty:
        return go.Figure()
    
    # Questions and their labels
    ai_questions = [
        ('Q20_ai_interaction_label', 'AI Increased Interaction'),
        ('Q22_ai_understand_label', 'AI Increased Understanding'), 
        ('Q23_ai_exp_label', 'AI Increased Experience')
    ]
    
    # Prepare data for grouped bar chart
    categories = ['YES!', 'yes', 'no', 'NO!']
    colors = {'YES!': '#2E8B57', 'yes': '#90EE90', 'no': '#FFB6C1', 'NO!': '#DC143C'}
    
    fig = go.Figure()
    
    x_positions = list(range(len(ai_questions)))
    bar_width = 0.2
    
    for i, category in enumerate(categories):
        values = []
        for question_col, _ in ai_questions:
            if question_col in ai_participants.columns:
                count = (ai_participants[question_col] == category).sum()
                values.append(count)
            else:
                values.append(0)
        
        fig.add_trace(go.Bar(
            name=category,
            x=[pos + i * bar_width for pos in x_positions],
            y=values,
            marker_color=colors[category],
            text=values,
            textposition='outside',
            width=bar_width,
            showlegend=True
        ))
    
    fig.update_layout(
        title={
            'text': f"AI Perception Comparison<br><sub>Based on {len(ai_participants)} participants who read about AI</sub>",
            'font': {'size': 16, 'family': 'Arial', 'color': '#2c3e50'},
            'x': 0.5
        },
        xaxis=dict(
            tickvals=[pos + 1.5 * bar_width for pos in x_positions],
            ticktext=[label for _, label in ai_questions],
            title="AI Perception Questions",
            tickfont={'size': 11, 'family': 'Arial', 'color': '#2c3e50'}
        ),
        yaxis=dict(
            title="Number of Responses",
            title_font={'size': 12, 'family': 'Arial', 'color': '#2c3e50'},
            tickfont={'size': 11, 'family': 'Arial', 'color': '#2c3e50'}
        ),
        barmode='group',
        height=600,
        legend=dict(
            title="Response",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font={'size': 11, 'family': 'Arial', 'color': '#2c3e50'}
        ),
        font={'size': 12, 'family': 'Arial', 'color': '#2c3e50'},
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def create_ai_summary_metrics(df):
    """Create summary metrics for AI perception"""
    
    ai_participants = df[df['Q19_read_ai'] == 1]
    total_ai = len(ai_participants)
    
    if total_ai == 0:
        return None, None, None, 0
    
    # Calculate positive responses (YES! + yes) for each question
    metrics = {}
    
    ai_cols = [
        ('Q20_ai_interaction', 'Interaction'),
        ('Q22_ai_understand', 'Understanding'),
        ('Q23_ai_exp', 'Experience')
    ]
    
    for col, label in ai_cols:
        if col in ai_participants.columns:
            positive = (ai_participants[col] <= 2).sum()  # YES! = 1, yes = 2
            percentage = (positive / total_ai * 100) if total_ai > 0 else 0
            metrics[label] = {'count': positive, 'percentage': percentage}
    
    return metrics, total_ai

# =====================================================
# COLLAB STATISTICAL ANALYSIS FUNCTION
# =====================================================
def run_kruskal_posthoc(group_df, questions):
    """
    Perform Kruskal-Wallis tests for specified questions and post-hoc Mann-Whitney U tests if significant.
    """
    from scipy.stats import kruskal, mannwhitneyu
    import itertools

    insights = []
    for question in questions:
        data = [group_df[group_df['Group'] == g][question].dropna() for g in group_df['Group'].unique()]
        if len(data) < 2 or any(len(d) == 0 for d in data):
            insights.append(f"Not enough data for {question}.")
            continue

        stat, p = kruskal(*data)
        insights.append(f"**{question}** Kruskal-Wallis H={stat:.3f}, p={p:.4f}")

        if p < 0.05:
            insights.append(f"Significant difference found in {question}, running post-hoc tests:")
            pairs = list(itertools.combinations(group_df['Group'].unique(), 2))
            for g1, g2 in pairs:
                d1 = group_df[group_df['Group'] == g1][question].dropna()
                d2 = group_df[group_df['Group'] == g2][question].dropna()
                if len(d1) == 0 or len(d2) == 0:
                    continue
                stat_u, p_u = mannwhitneyu(d1, d2, alternative='two-sided')
                p_corrected = p_u * len(pairs)  # Bonferroni correction
                insights.append(f"- {g1} vs {g2}: U={stat_u:.3f}, raw p={p_u:.4f}, corrected p={min(p_corrected,1):.4f}")
        else:
            insights.append(f"No significant difference found in {question}.")
    return insights
    """
    Perform Kruskal-Wallis tests for Q10–Q12 and post-hoc Mann-Whitney U tests if significant.
    """


import plotly.express as px

def create_boxplots(group_df, questions, title_prefix):
    """
    Create boxplots for the specified questions across groups.
    """
    fig_list = []
    for question in questions:
        if question not in group_df.columns:
            continue
        fig = px.box(group_df, x='Group', y=question, points='all',
                     color='Group', title=f"{title_prefix}: {question}",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_yaxes(autorange="reversed")  # Flip Y-axis
        fig.update_layout(height=400)
        fig_list.append(fig)
    return fig_list



# =====================================================
# AI IMPACT ANALYSIS FOR NEW SUBTAB
# =====================================================

def create_ai_impact_detailed_analysis(df):
    """Comprehensive analysis of AI impact on different player types"""
    
    # Filter AI participants
    ai_participants = df[df['Q19_read_ai'] == 1]
    non_ai_participants = df[df['Q19_read_ai'] != 1]
    
    if len(ai_participants) == 0:
        return None, "No AI participants found"
    
    # Compare AI vs Non-AI across multiple dimensions
    comparison_metrics = ['Q3_have_fun', 'Q6_enjoyed_fig', 'Q9_knew_solve', 
                         'Q10_fun_more', 'Q11_understand_more', 'Q12_enjoy_more']
    
    available_metrics = [m for m in comparison_metrics if m in df.columns]
    
    if len(available_metrics) == 0:
        return None, "No comparison metrics available"
    
    results = {}
    for metric in available_metrics:
        ai_scores = ai_participants[metric].dropna()
        non_ai_scores = non_ai_participants[metric].dropna()
        
        if len(ai_scores) > 0 and len(non_ai_scores) > 0:
            # Statistical test
            from scipy.stats import mannwhitneyu
            stat, p_value = mannwhitneyu(ai_scores, non_ai_scores, alternative='two-sided')
            
            # Effect size (Cohen's d approximation)
            ai_mean = ai_scores.mean()
            non_ai_mean = non_ai_scores.mean()
            pooled_std = np.sqrt((ai_scores.var() + non_ai_scores.var()) / 2)
            effect_size = (ai_mean - non_ai_mean) / pooled_std if pooled_std > 0 else 0
            
            results[metric] = {
                'ai_mean': ai_mean,
                'non_ai_mean': non_ai_mean,
                'ai_count': len(ai_scores),
                'non_ai_count': len(non_ai_scores),
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < 0.05,
                'direction': 'AI Higher' if ai_mean < non_ai_mean else 'Non-AI Higher'  # Lower scores = better (1=YES!, 4=NO!)
            }
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Score Comparison (Lower = Better)', 'Effect Sizes', 
                       'Statistical Significance', 'AI-Specific Responses'],
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'bar'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    if results:
        # 1. Score Comparison
        metrics_labels = [m.replace('_', ' ').title() for m in results.keys()]
        ai_means = [results[m]['ai_mean'] for m in results.keys()]
        non_ai_means = [results[m]['non_ai_mean'] for m in results.keys()]
        
        fig.add_trace(go.Bar(
            name='AI Group', 
            x=metrics_labels, 
            y=ai_means,
            marker_color='#FF7043',
            text=[f'{val:.2f}' for val in ai_means],
            textposition='outside'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            name='Non-AI Group', 
            x=metrics_labels, 
            y=non_ai_means, 
            marker_color='#1DB3B9',
            text=[f'{val:.2f}' for val in non_ai_means],
            textposition='outside'
        ), row=1, col=1)
        
        fig.update_yaxes(autorange="reversed")

        # 2. Effect Sizes
        effect_sizes = [results[m]['effect_size'] for m in results.keys()]
        effect_colors = ['#DC143C' if abs(es) > 0.5 else '#FFC107' if abs(es) > 0.2 else '#4CAF50' 
                        for es in effect_sizes]
        
        fig.add_trace(go.Bar(
            x=metrics_labels, 
            y=effect_sizes, 
            marker_color=effect_colors,
            name='Effect Size',
            text=[f'{es:.2f}' for es in effect_sizes],
            textposition='outside',
            showlegend=False
        ), row=1, col=2)
        
        # Add reference lines for effect size interpretation
        fig.add_hline(y=0.2, line_dash="dash", line_color="orange", opacity=0.5, row=1, col=2)
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5, row=1, col=2)
        fig.add_hline(y=-0.2, line_dash="dash", line_color="orange", opacity=0.5, row=1, col=2)
        fig.add_hline(y=-0.5, line_dash="dash", line_color="red", opacity=0.5, row=1, col=2)
        
        # 3. Statistical Significance
        p_values = [results[m]['p_value'] for m in results.keys()]
        significance_colors = ['#4CAF50' if p < 0.05 else '#FFC107' if p < 0.1 else '#DC143C' 
                              for p in p_values]
        
        fig.add_trace(go.Scatter(
            x=metrics_labels, 
            y=p_values, 
            mode='markers',
            marker=dict(size=15, color=significance_colors),
            name='P-values',
            text=[f'p={p:.3f}' for p in p_values],
            textposition='top center',
            showlegend=False
        ), row=2, col=1)
        
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange", row=2, col=1)
    
    # 4. AI-Specific Responses
    ai_specific_cols = ['Q20_ai_interaction', 'Q22_ai_understand', 'Q23_ai_exp']
    ai_specific_available = [col for col in ai_specific_cols if col in df.columns]
    
    if ai_specific_available:
        ai_specific_means = []
        ai_specific_labels = []
        
        for col in ai_specific_available:
            if col in ai_participants.columns:
                mean_val = ai_participants[col].mean()
                ai_specific_means.append(mean_val)
                label = col.replace('_', ' ').replace('ai ', 'AI ').title()
                ai_specific_labels.append(label)
        
        if ai_specific_means:
            fig.add_trace(go.Bar(
                x=ai_specific_labels, 
                y=ai_specific_means, 
                marker_color='#6A60A9',
                name='AI Impact',
                text=[f'{val:.2f}' for val in ai_specific_means],
                textposition='outside',
                showlegend=False
            ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Comprehensive AI Impact Analysis",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    
    # Update axes labels
    fig.update_yaxes(title_text="Score (1=YES!, 4=NO!)", row=1, col=1)
    fig.update_yaxes(title_text="Effect Size", row=1, col=2)
    fig.update_yaxes(title_text="P-value", row=2, col=1)
    fig.update_yaxes(title_text="AI Rating", row=2, col=2)
    
    return fig, results

def create_ai_impact_summary_table(results):
    """Create summary table of AI impact results"""
    
    if not results:
        return None
    
    summary_data = []
    for metric, data in results.items():
        summary_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'AI Group Mean': f"{data['ai_mean']:.2f}",
            'Non-AI Group Mean': f"{data['non_ai_mean']:.2f}",
            'Effect Size': f"{data['effect_size']:.3f}",
            'P-value': f"{data['p_value']:.4f}",
            'Significant': '✓' if data['significant'] else '✗',
            'Direction': data['direction']
        })
    
    return pd.DataFrame(summary_data)


# =====================================================
# MAIN APP
# =====================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎮 Rainbow Agents Survey Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # File upload option
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV File", type=['csv'], 
                                            help="Upload your survey CSV file")
    
    # Load data
    df = load_and_clean_data(uploaded_file)
    
    # Sidebar filters
    st.sidebar.header("📊 Filters & Options")
    
    # Grade filter
    if 'Q13_grade_label' in df.columns:
        grades = ['All'] + list(df['Q13_grade_label'].dropna().unique())
        selected_grade = st.sidebar.selectbox("Filter by Grade:", grades)
        
        if selected_grade != 'All':
            df = df[df['Q13_grade_label'] == selected_grade]
    
    # Gender filter
    if 'Q14_gender_label' in df.columns:
        genders = ['All'] + list(df['Q14_gender_label'].dropna().unique())
        selected_gender = st.sidebar.selectbox("Filter by Gender:", genders)
        
        if selected_gender != 'All':
            df = df[df['Q14_gender_label'] == selected_gender]
    
    
    
    
    # Analysis selection
    analysis_options = [
        "🏛️ Overview Dashboard",
        "🥁 Engagement Analysis", 
        "💡 Understanding & Performance",
        "🙌🏽 Collaboration Analysis",
        "🤖 AI Perception Analysis", 
        "🕶️ Demographics",
        "🔗 Correlations & Insights"
    ]
    
    selected_analysis = st.sidebar.radio("Choose Analysis:", analysis_options)
    
    # Display current filter info
    if selected_grade != 'All' or selected_gender != 'All':
        st.sidebar.markdown("### 🔍 Active Filters:")
        if selected_grade != 'All':
            st.sidebar.write(f"Grade: {selected_grade}")
        if selected_gender != 'All':
            st.sidebar.write(f"Gender: {selected_gender}")
        st.sidebar.write(f"Sample size: {len(df)} responses")
    
    # =====================================================
    # OVERVIEW DASHBOARD
    # =====================================================
    if selected_analysis == "🏛️ Overview Dashboard":
        st.header("🏛️ Overview Dashboard")
        
        # 4 key metrics using the AI Perception style
        col1, col2, col3, col4 = st.columns(4)
        
        # Clean dataset - ensure no extra rows
        total_responses = len(df)
        
        # Engagement: Q3 and Q6 positive (Likert 1 or 2)
        high_engagement = (df['Q3_have_fun'].isin([1,2]) & df['Q6_enjoyed_fig'].isin([1,2])).sum()
        engagement_percentage = (high_engagement / total_responses) * 100
        
        # Understood: Q9 positive (Likert 1 or 2)
        understood = df['Q9_knew_solve'].isin([1,2]).sum()
        understood_percentage = (understood / total_responses) * 100
        
        # Collaboration: Q10, Q11, Q12 all positive (Likert 1 or 2)
        collab_positive = df[df['Q10_fun_more'].isin([1,2]) & df['Q11_understand_more'].isin([1,2]) & df['Q12_enjoy_more'].isin([1,2])].shape[0]
        collab_percentage = (collab_positive / total_responses) * 100
        
        with col1:
            st.caption("Total Responses")
            st.subheader(f"**{total_responses}**")
        
        with col2:
            st.caption("High Engagement", help="Proportion of respondents with Q3 and Q6 positive divided by total.")
            st.subheader(f"**{high_engagement} ({engagement_percentage:.1f}%)**")
        
        with col3:
            st.caption("Understood Game",help="Proportion of respondents with Q9 positive divided by total.")
            st.subheader(f"**{understood} ({understood_percentage:.1f}%)**")
        
        with col4:
            st.caption("Positive Collaboration",help="Proportion of respondents with Q10, Q11, and Q12 all positive divided by total.")
            st.subheader(f"**{collab_positive} ({collab_percentage:.1f}%)**")

        # Quick Insights with highlighted box
        st.markdown("""
        <div class="insight-box">
            <h4>🔍 Quick Insights</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if 'Q3_have_fun_label' in df.columns:
                fun_chart = create_donut_chart(df, 'Q3_have_fun', 'Q3_have_fun_label', "Q3: I was having fun")
                st.plotly_chart(fun_chart, use_container_width=True)
        with col2:
            if 'Q6_enjoyed_fig_label' in df.columns:
                challenge_chart = create_donut_chart(df, 'Q6_enjoyed_fig', 'Q6_enjoyed_fig_label', "Q6: I enjoyed figuring things out")
                st.plotly_chart(challenge_chart, use_container_width=True)

    # =====================================================
    # ENGAGEMENT ANALYSIS
    # =====================================================
    
    elif selected_analysis == "🥁 Engagement Analysis":
        st.header("Engagement Analysis")
        
        tab1, tab2, tab3 = st.tabs(["🎭 Fun & Challenge", "🔗 Correlation", "📈 Combined Analysis"])
        
        with tab1:
            st.subheader("Player Engagement: Fun and Challenge")
                        # ADD THIS NEW SECTION:
            st.write("This chart shows positive responses (right) vs negative responses (left) from a center baseline.")
            
            # Choose which version you prefer:
            diverging_chart = create_diverging_engagement_percentage_chart(df)  # Count-based
            # OR
            # diverging_chart = create_diverging_engagement_percentage_chart(df)  # Percentage-based
            
            st.plotly_chart(diverging_chart, use_container_width=True)
            
            # Optional: Add insight box
            if 'Q3_have_fun_label' in df.columns and 'Q6_enjoyed_fig_label' in df.columns:
                q3_positive = (df['Q3_have_fun_label'].isin(['YES!', 'yes'])).sum()
                q3_total = df['Q3_have_fun_label'].notna().sum()
                q6_positive = (df['Q6_enjoyed_fig_label'].isin(['YES!', 'yes'])).sum()
                q6_total = df['Q6_enjoyed_fig_label'].notna().sum()
                
                q3_pct = (q3_positive / q3_total * 100) if q3_total > 0 else 0
                q6_pct = (q6_positive / q6_total * 100) if q6_total > 0 else 0
                
                st.markdown(f'''
                <div class="insight-box">
                <h4>💡 Engagement Insight</h4>
                <p><strong>{q3_pct:.1f}%</strong> had positive fun experiences, while <strong>{q6_pct:.1f}%</strong> enjoyed the challenges. 
                {'Both show strong engagement!' if min(q3_pct, q6_pct) > 60 else 'Mixed engagement patterns observed.'}</p>
                </div>
                ''', unsafe_allow_html=True)
       

            col1, col2 = st.columns(2)
            
            with col1:
                if 'Q3_have_fun_label' in df.columns:
                    fun_chart = create_ordered_likert_chart(df, 'Q3_have_fun', 'Q3_have_fun_label', 
                                                           "Q3: I was having fun")
                    st.plotly_chart(fun_chart, use_container_width=True)
            
            with col2:
                if 'Q6_enjoyed_fig_label' in df.columns:
                    challenge_chart = create_ordered_likert_chart(df, 'Q6_enjoyed_fig', 'Q6_enjoyed_fig_label', 
                                                                 "Q6: I enjoyed figuring things out")
                    st.plotly_chart(challenge_chart, use_container_width=True)
        
        
        with tab2:
            st.subheader("Did Players Have Fun While Embracing the Challenge?")

            # Correlation insights
            correlation = df['Q3_have_fun'].corr(df['Q6_enjoyed_fig'])
            st.markdown(f"""
            <div class="insight-box">
            <h4>💡 Insight</h4>
            <p>The correlation between having fun and enjoying challenges is <strong>{correlation:.3f}</strong>, 
            indicating a {'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.4 else 'weak'} 
            positive relationship.</p>
            </div>
            """, unsafe_allow_html=True)

            if all(col in df.columns for col in ['Q3_have_fun', 'Q6_enjoyed_fig']):
                corr_chart = create_correlation_heatmap(
                    df, ['Q3_have_fun', 'Q6_enjoyed_fig'], 
                    ['I was having fun', 'I enjoyed figuring things out'],
                    "Engagement Correlation"
                )
                st.plotly_chart(corr_chart, use_container_width=True)
                
                # Add grouped bar chart showing comparison
                grouped_chart = create_engagement_grouped_chart(df)
                st.plotly_chart(grouped_chart, use_container_width=True)
                
        
        with tab3:
            # Correlation heatmap/matrix
            if all(col in df.columns for col in ['Q3_have_fun', 'Q6_enjoyed_fig']):
                st.subheader("Engagement Correlation Matrix")
                
                # Create correlation matrix for engagement questions
                engagement_cols = ['Q3_have_fun', 'Q6_enjoyed_fig']
                if 'Q4_good_job' in df.columns:
                    engagement_cols.append('Q4_good_job')
                if 'Q9_knew_solve' in df.columns:
                    engagement_cols.append('Q9_knew_solve')
                
                available_cols = [col for col in engagement_cols if col in df.columns]
                
                if len(available_cols) >= 2:
                    col_labels = {
                        'Q3_have_fun': 'I was having fun',
                        'Q6_enjoyed_fig': 'I enjoyed figuring things out',
                        'Q4_good_job': 'I did a good job',
                        'Q9_knew_solve': 'I knew how to solve problems'
                    }
                    
                    labels = [col_labels[col] for col in available_cols]
                    
                    corr_matrix_chart = create_correlation_heatmap(df, available_cols, labels, 
                                                                  "Engagement & Performance Correlation Matrix")
                    st.plotly_chart(corr_matrix_chart, use_container_width=True)
                    
                    # Show correlation values in a table
                    corr_data = df[available_cols].corr()
                    corr_data.index = labels
                    corr_data.columns = labels
                    
                    st.subheader("Correlation Values")
                    st.dataframe(corr_data.round(3), use_container_width=True)
    
    # =====================================================
    # UNDERSTANDING & PERFORMANCE
    # =====================================================
    
    elif selected_analysis == "💡 Understanding & Performance":
        st.header("Understanding & Performance")
        st.subheader("🕹️ How well did players understand the game?")
            # Create two subtabs
        tab1, tab2 = st.tabs(["📊 Donut Charts", "🦁 Animal Strategy Flow"])
        
        with tab1:
            # First row: Q and Q
            row1_col1, row1_col2 = st.columns(2)
                    
            with row1_col1:
                if 'Q9_knew_solve_label' in df.columns:
                    understanding_chart = create_donut_chart(df, 'Q9_knew_solve', 'Q9_knew_solve_label', 
                                                        "Q9: I knew how to solve problems")
                    st.plotly_chart(understanding_chart, use_container_width=True)
            
            with row1_col2:
                if 'Q4_good_job_label' in df.columns:
                    performance_chart = create_donut_chart(df, 'Q4_good_job', 'Q4_good_job_label', 
                                                        "Q4: I did a good job")
                    st.plotly_chart(performance_chart, use_container_width=True)
            # Second row: Q9 and Q5
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                if 'Q5_do_cs' in df.columns:
                    # Create the donut chart for Q5
                    do_cs_chart = create_donut_chart(df, 'Q5_do_cs', 'Q5_do_cs_label', "Q5: I was doing computer science")
                    st.plotly_chart(do_cs_chart, use_container_width=True)
            
            with row2_col2:
                if 'Q7_difff_animals_label' in df.columns:
                    animals_chart = create_donut_chart(df, 'Q7_difff_animals', 'Q7_difff_animals_label', 
                                                    "Q7: Used different animals")
                    st.plotly_chart(animals_chart, use_container_width=True)
        with tab2:
            # ADD THE ANIMAL USAGE ANALYSIS HERE:
            st.markdown("---")  # Separator
            st.subheader("🦁 Animal Strategy Flow Analysis") 
            st.write("How does using different animals relate to garden perception and learning outcomes?")

            animal_sankey = create_animal_strategy_sankey(df)
            st.plotly_chart(animal_sankey, use_container_width=True)
            
            # Add insights text
            st.markdown("""
            **💡 Key Questions Explored:**
            - Do students who use different animals enjoy challenges more?
            - Is there a connection between animal variety and understanding the game?
            - How does collaboration type relate to animal usage strategies?
            """)



    # =====================================================
    # COLLABORATION ANALYSIS
    # =====================================================
    
    elif selected_analysis == "🙌🏽 Collaboration Analysis":
        st.header("Collaboration Analysis")
        
        tab1, tab2, tab3 = st.tabs(["👥 Who Played With", "🙌🏽 Playing Together", "🎨 Playing with Friends or Strangers"])
        
        with tab1:
            # Q2 multi-select chart
            st.subheader("Who Did You Play With? (Multi-Select)")
            if 'Q2_play_with' in df.columns:
                collab_chart = create_multiselect_collaboration_chart(df, 'Q2_play_with', 
                                                                    "Collaboration Types")
                st.plotly_chart(collab_chart, use_container_width=True)
        
        with tab2:
            # Playing together tab
            st.subheader("How did people feel about their experience playing with others?")

            if 'Q10_fun_more' in df.columns and 'Q11_understand_more' in df.columns and 'Q12_enjoy_more' in df.columns:
                def compute_positive_percentage(column):
                    positive = df[column].isin([1, 2]).sum()  # YES! and yes mapped to 1 and 2
                    total = df[column].notna().sum()
                    return (positive / total * 100) if total > 0 else 0

                fun_pct = compute_positive_percentage('Q10_fun_more')
                understand_pct = compute_positive_percentage('Q11_understand_more')
                enjoy_pct = compute_positive_percentage('Q12_enjoy_more')

                st.markdown(f"""
                <div class="insight-box">
                <h4> Overview </h4>
                <ul>
                    <li><strong>{fun_pct:.1f}%</strong> had fun playing with others.</li>
                    <li><strong>{understand_pct:.1f}%</strong> understood more with others.</li>
                    <li><strong>{enjoy_pct:.1f}%</strong> found playing with others made the game more enjoyable.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'Q10_fun_more_label' in df.columns:
                    fun_collab_chart = create_donut_chart(df, 'Q10_fun_more', 'Q10_fun_more_label', 
                                                        "Q10: Fun with others")
                    st.plotly_chart(fun_collab_chart, use_container_width=True)
            
            with col2:
                if 'Q11_understand_more_label' in df.columns:
                    understand_collab_chart = create_donut_chart(df, 'Q11_understand_more', 'Q11_understand_more_label', 
                                                            "Q11: Understanding with others")
                    st.plotly_chart(understand_collab_chart, use_container_width=True)
            
            with col3:
                if 'Q12_enjoy_more_label' in df.columns:
                    enjoy_collab_chart = create_donut_chart(df, 'Q12_enjoy_more', 'Q12_enjoy_more_label', 
                                                        "Q12: Enjoyment with others")
                    st.plotly_chart(enjoy_collab_chart, use_container_width=True)


                
        with tab3:
            st.subheader("Does it feel different to play with someone you know?")


            # Define subtabs under Detailed Views
            subtab1, subtab2, subtab3 = st.tabs([
                "📊 Percentage Breakdown",
                "📈 Collaboration Comparison",
                "🌊 Flow Diagram"

            ])            

            with subtab1:
                st.write("Percentage breakdown of responses by collaboration type")
                stacked_chart = create_collaboration_stacked_bars(df)
                st.plotly_chart(stacked_chart, use_container_width=True)

            
            with subtab2:
                # Add overall takeaway summary at the top
                st.markdown("""
                ### 📌 Overall Takeaways
                - Playing with someone familiar appears to enhance **fun and enjoyment**, but does not significantly affect **understanding**.
                - Differences were statistically significant for **Q10 (fun)** and **Q12 (enjoyment)**, with **corrected p-values below 0.05**.
                """)

                group_df_collab = prepare_collaboration_groups(df, include_questions=['Q10_fun_more', 'Q11_understand_more', 'Q12_enjoy_more'])
                group_df_collab = group_df_collab[group_df_collab['Group'].isin(['With Known', 'With Unknown'])]
                if group_df_collab is not None and not group_df_collab.empty:
                    boxplots2 = create_boxplots(group_df_collab, ['Q10_fun_more', 'Q11_understand_more', 'Q12_enjoy_more'], "2-Group")
                    for fig in boxplots2:
                        st.plotly_chart(fig, use_container_width=True)
                    results2 = run_kruskal_posthoc(group_df_collab, questions=['Q10_fun_more', 'Q11_understand_more', 'Q12_enjoy_more'])
                    for res in results2:
                        st.markdown(f"- {res}")
                else:
                    st.info("No valid data for 2-Group comparison.")
            
            
            
            with subtab3:
                st.write("Flow showing collaboration patterns (percentages within each group)")
                sankey_chart = create_collaboration_sankey_proportional(df)
                st.plotly_chart(sankey_chart, use_container_width=True)
        


    # =====================================================
    # AI PERCEPTIONS
    # =====================================================

    elif selected_analysis == "🤖 AI Perception Analysis":
        st.header("🤖 AI Perception Analysis")
        
        # Check if we have AI participants
        ai_participants = df[df['Q19_read_ai'] == 1] if 'Q19_read_ai' in df.columns else pd.DataFrame()
        total_participants = len(df)
        ai_count = len(ai_participants)
        
        # Summary metrics at top
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Participants", total_participants)
        
        with col2:
            st.metric("Read About AI", f"{ai_count} ({ai_count/total_participants*100:.1f}%)")
        
        with col3:
            if ai_count > 0:
                metrics, _ = create_ai_summary_metrics(df)
                if metrics and 'Interaction' in metrics:
                    interaction_positive = metrics['Interaction']['percentage']
                    st.metric("AI Improved Interaction", f"{interaction_positive:.1f}%")
        
        with col4:
            if ai_count > 0 and metrics and 'Understanding' in metrics:
                understanding_positive = metrics['Understanding']['percentage']
                st.metric("AI Improved Understanding", f"{understanding_positive:.1f}%")
        
        if ai_count == 0:
            st.warning("No participants read about AI in this dataset.")
            return
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Individual Questions","🔬 AI Impact Analysis","📈 Comparison View", "💡 Insights"])
        
        with tab1:
            st.subheader("Individual AI Perception Questions")
            
            # Three columns for the three donut charts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'Q20_ai_interaction_label' in df.columns:
                    interaction_chart = create_ai_donut_chart(
                        df, 'Q20_ai_interaction', 'Q20_ai_interaction_label',
                        "Q20: AI & Interaction",
                        "AI increased my interaction with the other player"
                    )
                    st.plotly_chart(interaction_chart, use_container_width=True)
            
            with col2:
                if 'Q22_ai_understand_label' in df.columns:
                    understand_chart = create_ai_donut_chart(
                        df, 'Q22_ai_understand', 'Q22_ai_understand_label',
                        "Q22: AI & Understanding", 
                        "AI increased my understanding of the game"
                    )
                    st.plotly_chart(understand_chart, use_container_width=True)
            
            with col3:
                if 'Q23_ai_exp_label' in df.columns:
                    experience_chart = create_ai_donut_chart(
                        df, 'Q23_ai_exp', 'Q23_ai_exp_label',
                        "Q23: AI & Experience",
                        "AI increased my gameplay experience"
                    )
                    st.plotly_chart(experience_chart, use_container_width=True)
        
            # NEW TAB 2: AI Impact Analysis
        with tab2:
            st.subheader("🔬 Detailed AI Impact Analysis")
            st.write("Statistical comparison of AI vs Non-AI participants across key metrics")
            
            # Create the analysis
            impact_fig, impact_results = create_ai_impact_detailed_analysis(df)
            
            if impact_fig is not None:
                st.plotly_chart(impact_fig, use_container_width=True)
                
                # Add summary table
                st.subheader("📋 Statistical Summary")
                summary_table = create_ai_impact_summary_table(impact_results)
                if summary_table is not None:
                    st.dataframe(summary_table, use_container_width=True)
                
                # Add interpretation guide
                st.markdown('''
                **📖 How to Read This Analysis:**
                - **Effect Size**: <0.2 = small, 0.2-0.5 = medium, >0.5 = large effect
                - **P-value**: <0.05 = statistically significant difference
                - **Scores**: Lower scores are better (1=YES!, 4=NO!)
                - **Colors**: Green=good, Yellow=moderate, Red=concerning
                ''')
                
                # Key insights
                if impact_results:
                    significant_effects = [m for m, r in impact_results.items() if r['significant']]
                    large_effects = [m for m, r in impact_results.items() if abs(r['effect_size']) > 0.5]
                    
                    if significant_effects or large_effects:
                        st.markdown("### 🎯 Key Findings")
                        if significant_effects:
                            st.write(f"**Statistically significant differences found in:** {', '.join([m.replace('_', ' ').title() for m in significant_effects])}")
                        if large_effects:
                            st.write(f"**Large effect sizes observed in:** {', '.join([m.replace('_', ' ').title() for m in large_effects])}")
                    else:
                        st.info("No statistically significant differences found between AI and Non-AI groups.")
            else:
                st.warning("Unable to perform AI impact analysis with current data.")
            
            
            
        with tab3:
            st.subheader("AI Perception Comparison")
            
            comparison_chart = create_ai_comparison_chart(df)
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Show detailed breakdown table
            st.subheader("Detailed Breakdown")
            
            ai_questions = [
                ('Q20_ai_interaction_label', 'AI Increased Interaction'),
                ('Q22_ai_understand_label', 'AI Increased Understanding'), 
                ('Q23_ai_exp_label', 'AI Increased Experience')
            ]
            
            breakdown_data = []
            for question_col, question_name in ai_questions:
                if question_col in ai_participants.columns:
                    counts = ai_participants[question_col].value_counts()
                    total = len(ai_participants)
                    
                    row_data = {'Question': question_name}
                    for response in ['YES!', 'yes', 'no', 'NO!']:
                        count = counts.get(response, 0)
                        percentage = (count / total * 100) if total > 0 else 0
                        row_data[response] = f"{count} ({percentage:.1f}%)"
                    
                    breakdown_data.append(row_data)
            
            if breakdown_data:
                breakdown_df = pd.DataFrame(breakdown_data)
                st.dataframe(breakdown_df, use_container_width=True)
        
        with tab4:
            st.subheader("💡 AI Perception Insights")
            
            if ai_count > 0:
                metrics, total_ai = create_ai_summary_metrics(df)
                
                if metrics:
                    # Generate insights
                    insights = []
                    
                    # Overall AI reception
                    positive_counts = []
                    for metric_name, metric_data in metrics.items():
                        positive_counts.append(metric_data['percentage'])
                    
                    avg_positive = sum(positive_counts) / len(positive_counts)
                    
                    if avg_positive > 70:
                        sentiment = "very positive"
                    elif avg_positive > 50:
                        sentiment = "generally positive"
                    elif avg_positive > 30:
                        sentiment = "mixed"
                    else:
                        sentiment = "generally negative"
                    
                    insights.append(f"**Overall AI Reception**: {sentiment} ({avg_positive:.1f}% average positive response)")
                    
                    # Individual question insights
                    for metric_name, metric_data in metrics.items():
                        percentage = metric_data['percentage']
                        count = metric_data['count']
                        
                        if percentage > 60:
                            tone = "majority of participants felt AI improved"
                        elif percentage > 40:
                            tone = "roughly half felt AI improved"
                        else:
                            tone = "minority felt AI improved"
                        
                        insights.append(f"**{metric_name}**: {tone} their {metric_name.lower()} ({count}/{total_ai} participants, {percentage:.1f}%)")
                    
                    # Display insights
                    for insight in insights:
                        st.markdown(f"- {insight}")
                    
                    # Add comparison with non-AI participants if applicable
                    st.markdown("---")
                    st.subheader("🔄 AI vs Non-AI Comparison")
                    
                    non_ai_participants = df[df['Q19_read_ai'] != 1] if 'Q19_read_ai' in df.columns else pd.DataFrame()
                    
                    if len(non_ai_participants) > 0:
                        st.write(f"**AI Group**: {ai_count} participants")
                        st.write(f"**Non-AI Group**: {len(non_ai_participants)} participants")
                        
                        # Compare engagement scores between groups
                        if 'engagement_score' in df.columns:
                            ai_engagement = ai_participants['engagement_score'].mean()
                            non_ai_engagement = non_ai_participants['engagement_score'].mean()
                            
                            engagement_diff = ai_engagement - non_ai_engagement
                            direction = "higher" if engagement_diff > 0 else "lower"
                            
                            st.write(f"**Engagement Comparison**: AI group has {direction} average engagement ({ai_engagement:.2f} vs {non_ai_engagement:.2f})")
                    
                else:
                    st.info("Unable to generate insights - missing AI perception data.")
            else:
                st.info("No AI participants found in the current filtered dataset.")


# =====================================================
# DEMOGRAPHICS
# =====================================================

    elif selected_analysis == "🕶️ Demographics":
        st.header("Demographics")

        # Grade Distribution (vertical layout)
        st.subheader("📚 Grade Distribution")
        if 'Q13_grade_label' in df.columns:
            grade_chart = create_vertical_bar_chart(df, 'Q13_grade', 'Q13_grade_label', 
                                                "Grade Distribution")
            st.plotly_chart(grade_chart, use_container_width=True)

        st.markdown("---")  # Separator

        # Gender Identity & Ethnicity Distribution (side by side)
        st.subheader("👤 Gender Identity & 🌍 Ethnicity Distribution")

        col1, col2 = st.columns(2)

        with col1:
            if 'Q14_gender_label' in df.columns:
                gender_chart = create_donut_chart(df, 'Q14_gender', 'Q14_gender_label', 
                                                "Gender Identity", chart_type="gender")
                st.plotly_chart(gender_chart, use_container_width=True)

        with col2:
            st.subheader("🌍 Ethnicity (Per Respondent)")
            ethnicity_chart = create_ethnicity_donut_chart(df)
            st.plotly_chart(ethnicity_chart, use_container_width=True)

        # 'Other' Ethnicity Summary Table
        st.subheader("📝 'Other' Ethnicity Details")
        other_ethnicity_summary = create_other_ethnicity_summary(df)
        if not other_ethnicity_summary.empty:
            st.dataframe(other_ethnicity_summary)
        else:
            st.info("No detailed 'Other' ethnicity responses found.")

        st.markdown("---")  # Separator


        
    
    # =====================================================
    # CORRELATIONS & INSIGHTS
    # =====================================================
    
    elif selected_analysis == "🔗 Correlations & Insights":
        st.header("Correlations & Insights")
        
        # Correlation matrix
        numeric_cols = ['Q3_have_fun', 'Q4_good_job', 'Q6_enjoyed_fig', 'Q9_knew_solve']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) >= 2:
            col_labels = {
                'Q3_have_fun': 'Having Fun',
                'Q4_good_job': 'Did Good Job', 
                'Q6_enjoyed_fig': 'Enjoyed Challenges',
                'Q9_knew_solve': 'Understood Game'
            }
            
            labels = [col_labels[col] for col in available_cols]
            
            corr_chart = create_correlation_heatmap(df, available_cols, labels, 
                                                  "Question Correlations")
            st.plotly_chart(corr_chart, use_container_width=True)
        
        # Key insights
        st.subheader("🔍 Key Insights")
        
        insights = []
        
        if all(col in df.columns for col in ['Q3_have_fun', 'Q6_enjoyed_fig']):
            engagement_high = ((df['Q3_have_fun'] <= 2) & (df['Q6_enjoyed_fig'] <= 2)).sum()
            insights.append(f"**High Engagement**: {engagement_high}/{len(df)} participants ({engagement_high/len(df)*100:.1f}%) had high engagement")
        
        if 'Q9_knew_solve' in df.columns:
            understood = (df['Q9_knew_solve'] <= 2).sum()
            insights.append(f"**Understanding**: {understood}/{len(df)} participants ({understood/len(df)*100:.1f}%) felt they understood the game")
        
        if 'collaboration_score' in df.columns:
            collab_positive = (df['collaboration_score'] <= 2).sum()
            insights.append(f"**Collaboration**: {collab_positive}/{len(df)} participants ({collab_positive/len(df)*100:.1f}%) had positive collaboration experiences")
        
        for insight in insights:
            st.markdown(f"- {insight}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Rainbow Agents Survey Analysis Dashboard - Built with Streamlit*")

if __name__ == "__main__":
    main()
