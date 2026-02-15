<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FitPredict-ML Project Documentation</title>
    <style>
        :root {
            --primary-color: #0366d6;
            --text-color: #24292e;
            --bg-color: #ffffff;
            --code-bg: #f6f8fa;
            --border-color: #e1e4e8;
            --accent-green: #2ea44f;
            --accent-orange: #d73a49;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--bg-color);
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
        }

        /* Header & Badges */
        header {
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }

        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            border-bottom: none;
        }

        .badges {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 4px;
            overflow: hidden;
        }

        .badge span:first-child {
            background-color: #555;
            color: white;
            padding: 4px 8px;
        }

        .badge span:last-child {
            color: white;
            padding: 4px 8px;
        }

        /* Colors for badges */
        .badge.status span:last-child { background-color: #e3b341; color: black; } /* Yellow */
        .badge.python span:last-child { background-color: #3572A5; } /* Blue */
        .badge.model span:last-child { background-color: #2ea44f; } /* Green */
        .badge.dataset span:last-child { background-color: #f66a0a; } /* Orange */

        /* Typography */
        h2 {
            margin-top: 40px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }

        h3 {
            margin-top: 25px;
        }

        p {
            margin-bottom: 15px;
        }

        /* Table Styling */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        th, td {
            border: 1px solid var(--border-color);
            padding: 10px;
            text-align: left;
        }

        th {
            background-color: var(--code-bg);
            font-weight: 600;
        }

        /* Status Pills in Table */
        .status-pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .status-completed { background-color: #dafbe1; color: #1a7f37; }
        .status-progress { background-color: #fff8c5; color: #9a6700; }
        .status-pending { background-color: #f6f8fa; color: #6e7781; }

        /* Code Blocks */
        pre {
            background-color: #1b1f23;
            color: #f0f6fc;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            font-size: 0.9rem;
        }

        code {
            background-color: rgba(27,31,35,0.05);
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: monospace;
            font-size: 90%;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
            color: inherit;
        }

        /* Directory Structure Tree */
        .tree-structure {
            background-color: var(--code-bg);
            border: 1px solid var(--border-color);
            padding: 15px;
            border-radius: 6px;
            font-family: monospace;
            white-space: pre;
            color: #24292e;
        }

        /* Footer */
        footer {
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            text-align: center;
            color: #586069;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>

<div class="container">

    <header>
        <h1>FitPredict-ML: Multi-Modal Protein Fitness Prediction</h1>
        
        <div class="badges">
            <div class="badge status"><span>Status</span><span>In Progress</span></div>
            <div class="badge python"><span>Python</span><span>3.10+</span></div>
            <div class="badge model"><span>Model</span><span>ESM-2 (650M)</span></div>
            <div class="badge dataset"><span>Dataset</span><span>ProteinGym</span></div>
        </div>

        <p>
            <strong>FitPredict-ML</strong> is a deep learning project designed to predict the fitness effects of protein mutations without the need for expensive wet-lab experiments. By integrating <strong>Sequence (ESM-2)</strong>, <strong>Structure (DSSP/AlphaFold)</strong>, and <strong>Evolutionary (MSA)</strong> data, this model aims to outperform single-modality baselines in predicting functional outcomes for drug discovery and protein engineering.
        </p>
    </header>

    <section>
        <h2>🚀 Project Progress & Status</h2>
        <p>We are currently in <strong>Week 2</strong> of the implementation plan.</p>

        <table>
            <thead>
                <tr>
                    <th>Stage</th>
                    <th>Component</th>
                    <th>Status</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Stage 1</strong></td>
                    <td>Data Preparation</td>
                    <td><span class="status-pill status-completed">✅ Completed</span></td>
                    <td>Cleaned & processed &beta;-lactamase (BLAT_ECOLX) dataset from ProteinGym.</td>
                </tr>
                <tr>
                    <td><strong>Stage 2</strong></td>
                    <td>Sequence Branch</td>
                    <td><span class="status-pill status-completed">✅ Completed</span></td>
                    <td>Extracted 1280-d embeddings using ESM-2 (650M) for 4,996 mutants.</td>
                </tr>
                <tr>
                    <td><strong>Stage 3</strong></td>
                    <td>Baseline Model</td>
                    <td><span class="status-pill status-progress">🔄 In Progress</span></td>
                    <td>Training Ridge Regression baseline to establish Sequence-only performance.</td>
                </tr>
                <tr>
                    <td><strong>Stage 4</strong></td>
                    <td>Structure Branch</td>
                    <td><span class="status-pill status-pending">⏳ Pending</span></td>
                    <td>Integration of secondary structure features (DSSP) & contact maps.</td>
                </tr>
                <tr>
                    <td><strong>Stage 5</strong></td>
                    <td>Fusion Network</td>
                    <td><span class="status-pill status-pending">⏳ Pending</span></td>
                    <td>Implementation of Attention mechanism to fuse modalities.</td>
                </tr>
            </tbody>
        </table>
    </section>

    <section>
        <h2>📂 Repository Structure</h2>
        <div class="tree-structure">
FitPredict-ML/
│
├── data/
│   ├── BLAT_ECOLX_Stiffler_2015.csv       <span style="color:#6a737d"># Raw DMS data (Source: ProteinGym)</span>
│   └── beta_lactamase_esm2_embeddings.npy <span style="color:#6a737d"># Pre-computed ESM-2 embeddings (Matrix: 4996x1280)</span>
│
├── scripts/
│   ├── extract_embeddings.py              <span style="color:#6a737d"># Script for ESM-2 feature extraction</span>
│   └── train_baseline.py                  <span style="color:#6a737d"># Script for Ridge Regression baseline</span>
│
├── results/
│   └── baseline_plot.png                  <span style="color:#6a737d"># Visualization of Predicted vs. True Fitness</span>
│
└── README.md                              <span style="color:#6a737d"># Project Documentation</span>
        </div>
    </section>

    <section>
        <h2>🛠️ Setup & Installation</h2>
        
        <h3>Prerequisites</h3>
        <ul>
            <li><strong>Hardware:</strong> NVIDIA GPU (Recommended: RTX 2050 or better) for inference.</li>
            <li><strong>Software:</strong> Python 3.10+, PyTorch, Fair-ESM.</li>
        </ul>

        <h3>Environment Setup</h3>
        <pre><code># Clone the repository
git clone https://github.com/yourusername/FitPredict-ML.git
cd FitPredict-ML

# Install dependencies
pip install torch fair-esm pandas scikit-learn scipy matplotlib tqdm</code></pre>
    </section>

    <section>
        <h2>💻 Usage & Implementation</h2>

        <h3>1. Data Preparation (Completed)</h3>
        <p>
            We utilize the <strong>Deep Mutational Scanning (DMS)</strong> dataset for Beta-lactamase (<code>BLAT_ECOLX</code>), provided by ProteinGym. This dataset contains:
        </p>
        <ul>
            <li><strong>Wild-type:</strong> The original functional protein.</li>
            <li><strong>Mutants:</strong> 4,996 single amino acid substitutions.</li>
            <li><strong>DMS_score:</strong> The ground-truth fitness value derived from experimental assays.</li>
        </ul>

        <h3>2. Feature Extraction (Completed)</h3>
        <p>We use the <strong>ESM-2 (t33_650M_UR50D)</strong> protein language model to generate embeddings.</p>
        <ul>
            <li><strong>Method:</strong> Mean Pooling over the sequence length.</li>
            <li><strong>Output:</strong> A <code>(N, 1280)</code> dimensional matrix saved as <code>.npy</code>.</li>
            <li><strong>Run Time:</strong> ~12 hours on CPU / ~30 mins on GPU.</li>
        </ul>
        <p>To re-run extraction:</p>
        <pre><code>python scripts/extract_embeddings.py</code></pre>

        <h3>3. Baseline Training (Current Step)</h3>
        <p>We are currently training a Ridge Regression model to establish a correlation baseline (Spearman &rho;).</p>
        <p>To run the baseline:</p>
        <pre><code>python scripts/train_baseline.py</code></pre>
    </section>

    <section>
        <h2>📊 Preliminary Results</h2>
        <ul>
            <li><strong>Dataset:</strong> <code>BLAT_ECOLX_Stiffler_2015</code></li>
            <li><strong>Embedding Source:</strong> ESM-2 (Layer 33)</li>
            <li><strong>Current Metric:</strong> <span style="color: #d73a49; font-weight: bold;">Pending</span> (Target: Spearman &rho; > 0.70)</li>
        </ul>
    </section>

    <section>
        <h2>📚 References</h2>
        <ol>
            <li><strong>ProteinGym:</strong> Benchmarks for Protein Mutation Prediction.</li>
            <li><strong>ESM-2:</strong> Evolutionary Scale Modeling (Meta AI).</li>
            <li><strong>Project Plan:</strong> "Multi-Modal Protein Fitness Prediction".</li>
        </ol>
    </section>

    <footer>
        <p>Maintained by Ansh &copy; 2026</p>
    </footer>

</div>

</body>
</html>