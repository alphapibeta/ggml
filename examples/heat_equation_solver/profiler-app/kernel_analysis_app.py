import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px


@st.cache_data
def load_data():
    df = pd.read_csv('./euler.csv')
    df['Block Size'] = df['Block Size'].astype(str)
    df['Kernel Type'] = df['Kernel Type'].astype(str)
    df['Primary Key'] = df['Kernel Type'] + '-' + df['Block Size']
    

    numeric_columns = [
        "GPU Execution Time (us)",
        "Achieved Occupancy (%)",
        "Memory Throughput (%)",
        "DRAM Throughput (%)",
        "SM Active Cycles (cycle)",
        "Compute (SM) Throughput (%)",
        "Theoretical Occupancy (%)",
        "Elapsed Cycles (cycle)",
        'Duration (us)'
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].replace({',': '', 'None': pd.NA, 'us': ''}, regex=True), errors='coerce')
    
    return df


df = load_data()


def normalize_and_plot(df, metric_name, k_metric, metric_type):

    if metric_type == 'min':
        top_k_metric_df = df.nsmallest(k_metric, metric_name).dropna(subset=[metric_name])
    elif metric_type == 'max':
        top_k_metric_df = df.nlargest(k_metric, metric_name).dropna(subset=[metric_name])


    st.header(f'Top {k_metric} Kernels with Block Sizes with Best "{metric_name}" ({metric_type.capitalize()}imized)')
    st.dataframe(top_k_metric_df)


    max_metrics = [
        "Achieved Occupancy (%)",
        "Memory Throughput (%)",
        "DRAM Throughput (%)",
        "SM Active Cycles (cycle)",
        "Compute (SM) Throughput (%)",
        "Theoretical Occupancy (%)",
        "Elapsed Cycles (cycle)"
    ]


    ultimate_df = top_k_metric_df[['Primary Key'] + max_metrics]


    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(ultimate_df[max_metrics])
    normalized_df = pd.DataFrame(normalized_values, columns=max_metrics)
    normalized_df['Primary Key'] = ultimate_df['Primary Key'].values


    def plot_metrics(data, max_metrics):
        fig, ax = plt.subplots(figsize=(30, 20))
        
        for metric in max_metrics:
            ax.plot(data['Primary Key'], data[metric], marker='o', label=f'{metric} (Normalized)', linewidth=2)

        ax.set_xlabel("Kernel Type + Block Size", fontsize=14)
        ax.set_ylabel("Normalized Metrics", fontsize=14)
        ax.set_title(f"Normalized Metrics for Top-k Kernels with {metric_type.capitalize()}imized '{metric_name}'", fontsize=16, fontweight='bold')
        ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=90)
        ax.grid(True)
        fig.tight_layout()
        
        st.pyplot(fig)


    plot_metrics(normalized_df, max_metrics)


min_metric_name = st.selectbox('Select a metric to minimize:', options=[
    "GPU Execution Time (us)",
    "Duration (us)"
], key='min_metric_name')

k_metric = st.number_input('Enter the number of top-k primary keys to display:', min_value=1, max_value=20, value=5, key='k_metric')

normalize_and_plot(df, min_metric_name, k_metric, metric_type='min')

st.header("Maximization Analysis")

max_metric_name = st.selectbox('Select a metric to maximize:', options=[
    "Achieved Occupancy (%)",
    "Memory Throughput (%)",
    "DRAM Throughput (%)",
    "SM Active Cycles (cycle)",
    "Compute (SM) Throughput (%)",
    "Theoretical Occupancy (%)",
    "Elapsed Cycles (cycle)"
], key='max_metric_name')

normalize_and_plot(df, max_metric_name, k_metric, metric_type='max')


st.header("Inter-Kernel Block Size Analysis")


inter_min_metric_names = st.multiselect('Select minimization metrics for inter-kernel analysis:', options=[
    "GPU Execution Time (us)",
    "Duration (us)"
], default=["GPU Execution Time (us)"])

inter_max_metric_names = st.multiselect('Select maximization metrics for inter-kernel analysis:', options=[
    "Achieved Occupancy (%)",
    "Memory Throughput (%)",
    "DRAM Throughput (%)",
    "SM Active Cycles (cycle)",
    "Compute (SM) Throughput (%)",
    "Theoretical Occupancy (%)",
    "Elapsed Cycles (cycle)"
], default=["Achieved Occupancy (%)"])


st.subheader("Top-K Kernel Selection for Inter-Kernel Analysis")


select_based_on = st.radio('Select top-k kernels based on:', options=['minimize', 'maximize'], index=0)


if select_based_on == 'minimize':
    top_metric_name = st.selectbox('Select a minimization metric:', options=inter_min_metric_names, key='top_metric_min')
else:
    top_metric_name = st.selectbox('Select a maximization metric:', options=inter_max_metric_names, key='top_metric_max')


top_k_for_inter = st.number_input('Select the number of top-k kernels for inter-kernel analysis:', min_value=1, max_value=100, value=50, key='top_k_inter')


def plot_block_size_impact_top_k(df, min_metrics, max_metrics, select_based_on, top_metric_name, top_k):

    if select_based_on == 'minimize':
        top_k_df = df.nsmallest(top_k, top_metric_name).dropna(subset=[top_metric_name])
    else:
        top_k_df = df.nlargest(top_k, top_metric_name).dropna(subset=[top_metric_name])

    block_size_groups = top_k_df.groupby('Primary Key')

    fig, ax1 = plt.subplots(figsize=(30, 20))


    colors_min = ['tab:blue', 'tab:cyan']
    for i, min_metric in enumerate(min_metrics):
        ax1.plot(block_size_groups[min_metric].mean().index, block_size_groups[min_metric].mean(), marker='o', color=colors_min[i % len(colors_min)], linestyle='-', linewidth=2, label=f'{min_metric} (Avg)')

    ax1.set_xlabel("Kernel Type + Block Size", fontsize=14)
    ax1.set_ylabel("Minimization Metrics", color='tab:blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='tab:blue')


    ax2 = ax1.twinx()
    colors_max = ['tab:red', 'tab:orange']
    for i, max_metric in enumerate(max_metrics):
        ax2.plot(block_size_groups[max_metric].mean().index, block_size_groups[max_metric].mean(), marker='o', color=colors_max[i % len(colors_max)], linestyle='-', linewidth=2, label=f'{max_metric} (Avg)')

    ax2.set_ylabel("Maximization Metrics", color='tab:red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle(f'Impact of Block Size on Selected Minimization and Maximization Metrics (Top-{top_k})', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='x', rotation=90)
    fig.tight_layout()
    ax1.grid(True)

    fig.legend(loc='upper right', bbox_to_anchor=(1.15, 1), prop={'size': 14})
    st.pyplot(fig)


plot_block_size_impact_top_k(df, inter_min_metric_names, inter_max_metric_names, select_based_on, top_metric_name, top_k_for_inter)




selected_metrics = [
    "GPU Execution Time (us)",
    "Achieved Occupancy (%)",
    "Memory Throughput (%)",
    "DRAM Throughput (%)",
    "SM Active Cycles (cycle)",
    "Compute (SM) Throughput (%)",
    "Theoretical Occupancy (%)",
    "Elapsed Cycles (cycle)",
    'Duration (us)'
]


st.subheader("Multiple Metrics Comparison")
selected_kernel = st.selectbox('Select a Kernel for Block Size Comparison:', df['Kernel Type'].unique(), key='selected_kernel')


kernel_df = df[df['Kernel Type'] == selected_kernel]

metric_1 = st.selectbox('Select the first metric:', selected_metrics, key='metric_1')
metric_2 = st.selectbox('Select the second metric:', selected_metrics, key='metric_2')

fig, ax1 = plt.subplots(figsize=(14, 8))

color = 'tab:blue'
ax1.set_xlabel('Block Size')
ax1.set_ylabel(metric_1, color=color)
ax1.plot(kernel_df['Block Size'], kernel_df[metric_1], color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)
plt.xticks(rotation=90)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel(metric_2, color=color)
ax2.plot(kernel_df['Block Size'], kernel_df[metric_2], color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
st.pyplot(fig)


st.subheader("Metric Explanations")
st.text("""
GPU Execution Time (us): Time taken by the GPU to execute the kernel.
Achieved Occupancy (%): Ratio of active warps on the GPU to the maximum number of possible active warps.
Memory Throughput (%): The amount of data read/written to/from memory.
DRAM Throughput (%): The throughput of data between GPU and DRAM.
SM Active Cycles (cycle): Number of cycles during which the Streaming Multiprocessor (SM) is active.
Compute (SM) Throughput (%): The amount of computational work done by the SMs.
Theoretical Occupancy (%): The ideal ratio of active warps to the maximum number.
Elapsed Cycles (cycle): Total cycles elapsed during kernel execution.
Duration (us): Total duration of the kernel execution.
""")


st.subheader("Occupancy Analysis")
fig, ax = plt.subplots(figsize=(14, 8))
sns.lineplot(x='Block Size', y='Achieved Occupancy (%)', data=df, ax=ax, marker='o', label='Achieved Occupancy')
sns.lineplot(x='Block Size', y='Theoretical Occupancy (%)', data=df, ax=ax, marker='o', label='Theoretical Occupancy')
ax.set_title('Achieved vs Theoretical Occupancy', fontsize=16)
ax.set_xlabel("Block Size", fontsize=14)
ax.set_ylabel("Occupancy (%)", fontsize=14)
plt.xticks(rotation=90)
st.pyplot(fig)


st.subheader("Performance Efficiency")
df['Performance Efficiency'] = df['Compute (SM) Throughput (%)'] / df['Memory Throughput (%)']
fig, ax = plt.subplots(figsize=(14, 8))
sns.lineplot(x='Block Size', y='Performance Efficiency', data=df, ax=ax, marker='o')
ax.set_title('Performance Efficiency (Compute/Memory Throughput) vs Block Size', fontsize=16)
ax.set_xlabel("Block Size", fontsize=14)
ax.set_ylabel("Performance Efficiency", fontsize=14)
plt.xticks(rotation=90)
st.pyplot(fig)


st.subheader("Parallel Coordinates Plot")
fig = px.parallel_coordinates(df, color="GPU Execution Time (us)", dimensions=selected_metrics)
st.plotly_chart(fig)


st.subheader("Optimal Block Size Suggestion")
optimal_kernel_block = df.loc[df['GPU Execution Time (us)'].idxmin()]
optimal_block_size = optimal_kernel_block['Block Size']
optimal_kernel_name = optimal_kernel_block['Kernel Type']
st.write(f"The optimal block size for the lowest GPU Execution Time is: **{optimal_block_size}** with Kernel Type: **{optimal_kernel_name}**")
