<%@ page session="true" %>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Based Crop Market Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f7f9;
        }

        .header {
            background-color: #28a745;
            color: white;
            padding: 20px;
            text-align: center;
        }

        .header nav {
            display: flex;
            justify-content: center;
        }

        .header nav a {
            margin: 0 10px;
            color: white;
            text-decoration: none;
            display: inline-block;
        }

        .main {
            padding: 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }

        .heading {
            text-align: center;
            width: 100%;
            padding-bottom: 20px;
        }

        .section {
            background-color: white;
            padding: 20px;
            margin: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            flex: 1;
            min-width: 300px;
            max-width: 600px;
        }

        .section h2 {
            margin-top: 0;
            color: #28a745;
        }

        .forecast-chart {
            height: 300px;
            background-color: #f0f0f0;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #888;
            margin-bottom: 20px;
        }

        .alerts {
            color: red;
            list-style: none;
            padding-left: 0;
        }

        .alerts li {
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f8d7da;
            border-left: 5px solid #f5c6cb;
        }

        .market-trends {
            list-style: none;
            padding-left: 0;
        }

        .market-trends li {
            padding: 10px;
            margin-bottom: 10px;
            background-color: #e2e3e5;
            border-left: 5px solid #d6d8db;
        }

        .crop-option {
            display: flex;
            align-items: center;
        }

        .crop-option img {
            width: 20px;
            height: 20px;
            margin-right: 8px;
        }

        @media (max-width: 768px) {
            .main {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>

    <!-- Header -->
    <div class="header">
        <nav>
            <a href="#">Dashboard</a>
<!--            <a href="#">Market Pricing Prediction</a>-->
            <a href="about.html">About Us</a>
            <!--<a href="#">Recommendations</a>-->
            <a href="index.html">Logout</a>
        </nav>
    </div>

    <!-- Main Content Area -->
    <div class="main">
        <div class="heading">
            <h1>Welcome, <%= session.getAttribute("email") %></h1>
        </div>

        <!-- Price Prediction Chart -->
        <div class="section">
            <h2>Crop Price Prediction</h2>
            <div class="forecast-chart">
                <p>Price Chart (Maize, Wheat, etc.)</p>
            </div>
            <form id="cropForm" action="price.jsp" method="post">
                <input type="hidden" name="selectedCrop" id="selectedCrop">
                <select id="crops" onchange="navigateToPricePage()">
                    <option>Select Crop</option>
                    <option value="maize" class="crop-option">
                        <img src="images/maize.png" alt="Maize"> Maize
                    </option>
                    <option value="wheat" class="crop-option">
                        <img src="images/wheat.png" alt="Wheat"> Wheat
                    </option>
                    <option value="soybeans" class="crop-option">
                        <img src="images/soybeans.png" alt="Soybeans"> Soybeans
                    </option>
                </select>
            </form>
        </div>

        <!-- Real-Time Alerts -->
        <div class="section">
            <h2>Real-Time Price Alerts</h2>
            <ul class="alerts">
                <li>Maize price expected to rise by 10% next week.</li>
                <li>Wheat market dropping by 5%, consider selling now.</li>
            </ul>
        </div>

        <!-- Market Trends -->
        <div class="section">
            <h2>Market Trends</h2>
            <ul class="market-trends">
                <li>High demand for Maize in Limpopo region.</li>
                <li>Supply chain disruption affecting soybean prices.</li>
            </ul>
        </div>

        <!-- Recommendations -->
        <div class="section">
            <h2>Recommendations</h2>
            <p>Sell Maize in 2 weeks to maximize profit.</p>
            <p>Store Wheat for another 10 days for higher prices.</p>
        </div>

    </div>

    <script>
        function navigateToPricePage() {
            let crop = document.getElementById("crops").value;
            if (crop !== "Select Crop") {
                document.getElementById("selectedCrop").value = crop;
                document.getElementById("cropForm").submit();
            }
        }
    </script>
</body>
</html>
