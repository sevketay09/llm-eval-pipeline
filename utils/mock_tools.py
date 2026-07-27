"""
Mock Tool Execution Environment
Modellerin function calling yeteneklerini dinamik olarak test etmek için
gerçek tool fonksiyonlarını simüle eden mock environment.
"""
import ast
import json
import operator
import random
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime, timedelta

_SAFE_CALC_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_calc_eval(node):
    """Evaluate a numeric-only AST node. Raises ValueError/TypeError on anything
    that isn't a number, a unary +/-, or one of the arithmetic operators above —
    no names, calls, attributes, subscripts, comparisons, or string literals,
    so this can't be used as an eval()-based sandbox-escape or code-exec gadget.
    """
    if isinstance(node, ast.Expression):
        return _safe_calc_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_CALC_OPERATORS:
        return _SAFE_CALC_OPERATORS[type(node.op)](_safe_calc_eval(node.left), _safe_calc_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_CALC_OPERATORS:
        return _SAFE_CALC_OPERATORS[type(node.op)](_safe_calc_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def safe_calculate(expression: str):
    """Parse and evaluate a plain arithmetic expression (+ - * / // % ** and parens)
    without falling back to eval()/exec() on untrusted input."""
    parsed = ast.parse(expression, mode="eval")
    return _safe_calc_eval(parsed)


class MockToolEnvironment:
    """Mock tool execution environment for testing function calling"""
    
    def __init__(self, error_simulation_config: Optional[Dict[str, Any]] = None):
        self.tools = {}
        self.execution_history = []
        self.error_simulation_config = error_simulation_config or {}
        self.call_count = {}  # Track call counts for retry testing
        self._register_default_tools()
        self._register_error_prone_tools()
    
    def _register_default_tools(self):
        """Register default mock tools"""
        
        # Weather API
        self.register_tool(
            name="get_weather",
            function=self._mock_get_weather,
            description="Hava durumu bilgisi getirir",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Şehir adı"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Sıcaklık birimi (opsiyonel)"}
                },
                "required": ["city"]
            }
        )
        
        # Exchange Rate API
        self.register_tool(
            name="get_exchange_rate",
            function=self._mock_get_exchange_rate,
            description="Döviz kurunu getirir",
            parameters={
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string", "description": "Kaynak para birimi (USD, EUR, etc.)"},
                    "to_currency": {"type": "string", "description": "Hedef para birimi (TRY, USD, etc.)"}
                },
                "required": ["from_currency", "to_currency"]
            }
        )
        
        # Calculator
        self.register_tool(
            name="calculate",
            function=self._mock_calculate,
            description="Matematiksel hesaplama yapar",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Hesaplanacak matematiksel ifade"},
                    "precision": {"type": "integer", "description": "Sonuç hassasiyeti (ondalık basamak sayısı)"}
                },
                "required": ["expression"]
            }
        )
        
        # Database Query
        self.register_tool(
            name="query_database",
            function=self._mock_query_database,
            description="Veritabanından bilgi çeker",
            parameters={
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Tablo adı"},
                    "filters": {"type": "object", "description": "Filtreler"},
                    "limit": {"type": "integer", "description": "Maksimum sonuç sayısı"}
                },
                "required": ["table"]
            }
        )
        
        # Send Email
        self.register_tool(
            name="send_email",
            function=self._mock_send_email,
            description="E-posta gönderir",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Alıcı e-posta adresi"},
                    "subject": {"type": "string", "description": "E-posta konusu"},
                    "body": {"type": "string", "description": "E-posta içeriği"}
                },
                "required": ["to", "subject", "body"]
            }
        )
        
        # Get Stock Price
        self.register_tool(
            name="get_stock_price",
            function=self._mock_get_stock_price,
            description="Hisse senedi fiyatını getirir",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Hisse senedi sembolü"},
                    "date": {"type": "string", "description": "Tarih (YYYY-MM-DD formatında, opsiyonel)"}
                },
                "required": ["symbol"]
            }
        )
        
        # Get Flight Info
        self.register_tool(
            name="get_flight_info",
            function=self._mock_get_flight_info,
            description="Uçuş bilgilerini getirir",
            parameters={
                "type": "object",
                "properties": {
                    "from_city": {"type": "string", "description": "Kalkış şehri"},
                    "to_city": {"type": "string", "description": "Varış şehri"},
                    "date": {"type": "string", "description": "Uçuş tarihi (YYYY-MM-DD)"}
                },
                "required": ["from_city", "to_city", "date"]
            }
        )
        
        # Book Hotel
        self.register_tool(
            name="book_hotel",
            function=self._mock_book_hotel,
            description="Otel rezervasyonu yapar",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Şehir"},
                    "checkin": {"type": "string", "description": "Giriş tarihi (YYYY-MM-DD)"},
                    "checkout": {"type": "string", "description": "Çıkış tarihi (YYYY-MM-DD)"},
                    "guests": {"type": "integer", "description": "Kişi sayısı"}
                },
                "required": ["city", "checkin", "checkout"]
            }
        )
        
        # Get Restaurant Recommendations
        self.register_tool(
            name="get_restaurant_recommendations",
            function=self._mock_get_restaurant_recommendations,
            description="Restoran önerileri getirir",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Şehir"},
                    "cuisine": {"type": "string", "description": "Mutfak türü (opsiyonel)"},
                    "price_range": {"type": "string", "enum": ["budget", "moderate", "expensive"], "description": "Fiyat aralığı"}
                },
                "required": ["city"]
            }
        )
        
        # Search Flights
        self.register_tool(
            name="search_flights",
            function=self._mock_search_flights,
            description="Uçuş araması yapar",
            parameters={
                "type": "object",
                "properties": {
                    "from_city": {"type": "string", "description": "Kalkış şehri"},
                    "to_city": {"type": "string", "description": "Varış şehri"},
                    "date": {"type": "string", "description": "Uçuş tarihi (YYYY-MM-DD, opsiyonel)"}
                },
                "required": ["from_city", "to_city"]
            }
        )
        
        # Book Flight
        self.register_tool(
            name="book_flight",
            function=self._mock_book_flight,
            description="Uçuş rezervasyonu yapar",
            parameters={
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string", "description": "Uçuş ID'si"},
                    "passenger_name": {"type": "string", "description": "Yolcu adı"},
                    "seat_preference": {"type": "string", "description": "Koltuk tercihi (window/aisle)"}
                },
                "required": ["flight_id", "passenger_name"]
            }
        )
        
        # Search Product
        self.register_tool(
            name="search_product",
            function=self._mock_search_product,
            description="Ürün araması yapar",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Aranacak ürün"},
                    "category": {"type": "string", "description": "Ürün kategorisi (opsiyonel)"},
                    "max_price": {"type": "number", "description": "Maksimum fiyat"}
                },
                "required": ["query"]
            }
        )
        
        # Check Calendar
        self.register_tool(
            name="check_calendar",
            function=self._mock_check_calendar,
            description="Takvimde boş saatleri kontrol eder",
            parameters={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Tarih (YYYY-MM-DD)"},
                    "time_range": {"type": "string", "description": "Saat aralığı (örn: 09:00-17:00)"}
                },
                "required": ["date"]
            }
        )
        
        # Translate Text
        self.register_tool(
            name="translate_text",
            function=self._mock_translate_text,
            description="Metni çevirir",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Çevrilecek metin"},
                    "source_lang": {"type": "string", "description": "Kaynak dil kodu (tr, en, etc.)"},
                    "target_lang": {"type": "string", "description": "Hedef dil kodu"}
                },
                "required": ["text", "target_lang"]
            }
        )
    
    def register_tool(self, name: str, function: Callable, description: str, parameters: Dict[str, Any]):
        """Register a new tool"""
        self.tools[name] = {
            "function": function,
            "description": description,
            "parameters": parameters
        }
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given arguments
        
        Returns:
            {
                "success": bool,
                "result": any or None,
                "error": str or None,
                "error_type": str or None (for error recovery testing)
            }
        """
        # Track call count for this tool
        if tool_name not in self.call_count:
            self.call_count[tool_name] = 0
        self.call_count[tool_name] += 1
        
        # Record execution
        execution_record = {
            "tool_name": tool_name,
            "arguments": arguments,
            "timestamp": datetime.now().isoformat(),
            "attempt_number": self.call_count[tool_name]
        }
        
        if tool_name not in self.tools:
            execution_record["success"] = False
            execution_record["error"] = f"Tool '{tool_name}' not found"
            execution_record["error_type"] = "tool_not_found"
            self.execution_history.append(execution_record)
            return execution_record
        
        # Check for simulated errors
        if self._should_simulate_error(tool_name, self.call_count[tool_name]):
            error_type, error_msg = self._get_simulated_error(tool_name)
            execution_record["success"] = False
            execution_record["result"] = None
            execution_record["error"] = error_msg
            execution_record["error_type"] = error_type
            self.execution_history.append(execution_record)
            return execution_record
        
        try:
            tool_func = self.tools[tool_name]["function"]
            result = tool_func(**arguments)
            
            execution_record["success"] = True
            execution_record["result"] = result
            execution_record["error"] = None
            execution_record["error_type"] = None
            
        except Exception as e:
            execution_record["success"] = False
            execution_record["result"] = None
            execution_record["error"] = str(e)
            execution_record["error_type"] = "execution_error"
        
        self.execution_history.append(execution_record)
        return execution_record
    
    def _should_simulate_error(self, tool_name: str, attempt_number: int) -> bool:
        """Check if we should simulate an error for this tool call"""
        if tool_name not in self.error_simulation_config:
            return False
        
        config = self.error_simulation_config[tool_name]
        
        # Check failure rate
        if "failure_rate" in config:
            if random.random() < config["failure_rate"]:
                return True
        
        # Check if should fail on specific attempts
        if "fail_on_attempts" in config:
            if attempt_number in config["fail_on_attempts"]:
                return True
        
        # Check if should fail until attempt N
        if "fail_until_attempt" in config:
            if attempt_number < config["fail_until_attempt"]:
                return True
        
        return False
    
    def _get_simulated_error(self, tool_name: str) -> tuple[str, str]:
        """Get simulated error type and message"""
        config = self.error_simulation_config.get(tool_name, {})
        error_type = config.get("error_type", "transient_error")
        
        error_messages = {
            "transient_error": "Geçici bir hata oluştu. Lütfen tekrar deneyin.",
            "rate_limit": "Rate limit exceeded. Please try again in a few seconds.",
            "timeout": "İşlem zaman aşımına uğradı.",
            "service_unavailable": "Servis şu anda kullanılamıyor.",
            "invalid_parameter": "Geçersiz parametre.",
            "authentication_error": "Kimlik doğrulama hatası.",
            "permission_denied": "Bu işlem için yetkiniz yok."
        }
        
        error_msg = config.get("error_message", error_messages.get(error_type, "Bilinmeyen hata"))
        
        return error_type, error_msg
    
    def _register_error_prone_tools(self):
        """Register tools that are specifically designed to fail for testing error recovery"""
        
        # Unreliable Weather API
        self.register_tool(
            name="unreliable_weather_api",
            function=self._mock_unreliable_weather,
            description="Güvenilir olmayan hava durumu API'si (test için)",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Şehir adı"}
                },
                "required": ["city"]
            }
        )
        
        # Flaky Database
        self.register_tool(
            name="flaky_database",
            function=self._mock_flaky_database,
            description="Ara sıra hata veren veritabanı (test için)",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL sorgusu"}
                },
                "required": ["query"]
            }
        )
        
        # Rate limited API
        self.register_tool(
            name="rate_limited_api",
            function=self._mock_rate_limited_api,
            description="Rate limit'li API (test için)",
            parameters={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "API endpoint"}
                },
                "required": ["endpoint"]
            }
        )
    
    def _mock_unreliable_weather(self, city: str) -> Dict[str, Any]:
        """Unreliable weather API that sometimes fails"""
        # This will be controlled by error_simulation_config
        return self._mock_get_weather(city)
    
    def _mock_flaky_database(self, query: str) -> Dict[str, Any]:
        """Flaky database that sometimes fails"""
        return {
            "query": query,
            "results": [{"id": 1, "data": "sample"}],
            "count": 1
        }
    
    def _mock_rate_limited_api(self, endpoint: str) -> Dict[str, Any]:
        """Rate limited API"""
        return {
            "endpoint": endpoint,
            "data": {"message": "Success"},
            "rate_limit_remaining": random.randint(0, 100)
        }
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get tool execution history"""
        return self.execution_history
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history = []
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions"""
        tool_defs = []
        for name, tool_info in self.tools.items():
            tool_defs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_info["description"],
                    "parameters": tool_info["parameters"]
                }
            })
        return tool_defs
    
    # ==================== Mock Tool Implementations ====================
    
    def _mock_get_weather(self, city: str, unit: str = "celsius") -> Dict[str, Any]:
        """Mock weather API"""
        weather_conditions = ["Güneşli", "Parçalı bulutlu", "Bulutlu", "Yağmurlu", "Karlı", "Fırtınalı"]
        temp = random.randint(5, 35) if unit == "celsius" else random.randint(41, 95)
        
        return {
            "city": city,
            "temperature": temp,
            "unit": unit,
            "condition": random.choice(weather_conditions),
            "humidity": random.randint(30, 90),
            "wind_speed": random.randint(5, 40)
        }
    
    def _mock_get_exchange_rate(self, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """Mock exchange rate API"""
        # Mock rates (approximations)
        rates = {
            ("USD", "TRY"): 32.5,
            ("EUR", "TRY"): 35.2,
            ("GBP", "TRY"): 41.3,
            ("USD", "EUR"): 0.92,
            ("EUR", "USD"): 1.08,
            ("TRY", "USD"): 0.031
        }
        
        rate = rates.get((from_currency, to_currency))
        if rate is None:
            # Calculate inverse if exists
            inverse_rate = rates.get((to_currency, from_currency))
            if inverse_rate:
                rate = 1.0 / inverse_rate
            else:
                rate = 1.0  # fallback
        
        return {
            "from": from_currency,
            "to": to_currency,
            "rate": rate,
            "timestamp": datetime.now().isoformat()
        }
    
    def _mock_calculate(self, expression: str, precision: int = 2) -> Dict[str, Any]:
        """Mock calculator"""
        try:
            result = safe_calculate(expression)
            return {
                "expression": expression,
                "result": round(result, precision),
                "precision": precision
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": f"Calculation error: {str(e)}"
            }
    
    def _mock_query_database(self, table: str, filters: Optional[Dict] = None, limit: int = 10) -> Dict[str, Any]:
        """Mock database query"""
        # Generate mock data
        mock_data = []
        for i in range(min(limit, 5)):
            mock_data.append({
                "id": i + 1,
                "name": f"Record_{i+1}",
                "value": random.randint(100, 1000)
            })
        
        return {
            "table": table,
            "filters": filters,
            "count": len(mock_data),
            "results": mock_data
        }
    
    def _mock_send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        """Mock email sending"""
        return {
            "status": "sent",
            "to": to,
            "subject": subject,
            "message_id": f"msg_{random.randint(10000, 99999)}",
            "timestamp": datetime.now().isoformat()
        }
    
    def _mock_get_stock_price(self, symbol: str, date: Optional[str] = None) -> Dict[str, Any]:
        """Mock stock price API"""
        base_prices = {
            "AAPL": 180.0,
            "GOOGL": 140.0,
            "MSFT": 380.0,
            "THYAO": 350.0,
            "GARAN": 120.0
        }
        
        base_price = base_prices.get(symbol.upper(), 100.0)
        # Add some random variation
        price = base_price + random.uniform(-10, 10)
        
        return {
            "symbol": symbol.upper(),
            "price": round(price, 2),
            "currency": "USD" if symbol.upper() not in ["THYAO", "GARAN"] else "TRY",
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "change": round(random.uniform(-5, 5), 2)
        }
    
    def _mock_get_flight_info(self, from_city: str, to_city: str, date: str) -> Dict[str, Any]:
        """Mock flight information"""
        airlines = ["Turkish Airlines", "Pegasus", "AnadoluJet"]
        
        return {
            "from": from_city,
            "to": to_city,
            "date": date,
            "flights": [
                {
                    "airline": random.choice(airlines),
                    "flight_number": f"TK{random.randint(100, 999)}",
                    "departure": f"{random.randint(6, 22):02d}:{random.randint(0, 59):02d}",
                    "arrival": f"{random.randint(8, 23):02d}:{random.randint(0, 59):02d}",
                    "price": random.randint(500, 3000),
                    "available_seats": random.randint(5, 50)
                }
                for _ in range(3)
            ]
        }
    
    def _mock_book_hotel(self, city: str, checkin: str, checkout: str, guests: int = 2) -> Dict[str, Any]:
        """Mock hotel booking"""
        hotels = ["Hilton", "Marriott", "Sheraton", "Radisson"]
        
        return {
            "status": "confirmed",
            "hotel": random.choice(hotels),
            "city": city,
            "checkin": checkin,
            "checkout": checkout,
            "guests": guests,
            "confirmation_number": f"BK{random.randint(100000, 999999)}",
            "total_price": random.randint(1000, 5000),
            "currency": "TRY"
        }
    
    def _mock_get_restaurant_recommendations(self, city: str, cuisine: Optional[str] = None, 
                                            price_range: str = "moderate") -> Dict[str, Any]:
        """Mock restaurant recommendations"""
        restaurants = [
            {"name": "Lezzet Durağı", "cuisine": "Türk", "rating": 4.5},
            {"name": "Deniz Restaurant", "cuisine": "Balık", "rating": 4.7},
            {"name": "Pizza Palace", "cuisine": "İtalyan", "rating": 4.3},
            {"name": "Sushi Bar", "cuisine": "Japon", "rating": 4.6}
        ]
        
        # Filter by cuisine if specified
        if cuisine:
            restaurants = [r for r in restaurants if r["cuisine"].lower() == cuisine.lower()]
        
        return {
            "city": city,
            "cuisine": cuisine,
            "price_range": price_range,
            "recommendations": random.sample(restaurants, min(len(restaurants), 3))
        }
    
    def _mock_search_flights(self, from_city: str, to_city: str, date: Optional[str] = None) -> Dict[str, Any]:
        """Mock flight search"""
        airlines = ["Turkish Airlines", "Pegasus", "AnadoluJet", "Lufthansa", "British Airways"]
        flight_date = date or (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        
        flights = []
        for i in range(random.randint(3, 6)):
            departure_hour = random.randint(6, 22)
            duration_hours = random.randint(2, 12)
            arrival_hour = (departure_hour + duration_hours) % 24
            
            flights.append({
                "flight_id": f"FL{random.randint(1000, 9999)}",
                "airline": random.choice(airlines),
                "from": from_city,
                "to": to_city,
                "departure": f"{flight_date} {departure_hour:02d}:{random.randint(0, 59):02d}",
                "arrival": f"{flight_date} {arrival_hour:02d}:{random.randint(0, 59):02d}",
                "duration": f"{duration_hours}h {random.randint(0, 59)}m",
                "price": random.randint(800, 5000),
                "currency": "TRY",
                "available_seats": random.randint(5, 100),
                "stops": random.choice([0, 1])
            })
        
        return {
            "from_city": from_city,
            "to_city": to_city,
            "date": flight_date,
            "results_count": len(flights),
            "flights": flights
        }
    
    def _mock_book_flight(self, flight_id: str, passenger_name: str, seat_preference: Optional[str] = None) -> Dict[str, Any]:
        """Mock flight booking"""
        return {
            "status": "confirmed",
            "booking_id": f"BK{random.randint(100000, 999999)}",
            "flight_id": flight_id,
            "passenger_name": passenger_name,
            "seat": f"{random.randint(1, 30)}{random.choice(['A', 'B', 'C', 'D', 'E', 'F'])}",
            "seat_preference": seat_preference or "window",
            "confirmation_code": f"{''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789', k=6))}",
            "timestamp": datetime.now().isoformat()
        }
    
    def _mock_search_product(self, query: str, category: Optional[str] = None, max_price: Optional[float] = None) -> Dict[str, Any]:
        """Mock product search"""
        products = [
            {"name": f"{query} - Model A", "price": random.randint(500, 5000), "rating": round(random.uniform(3.5, 5.0), 1)},
            {"name": f"{query} - Premium", "price": random.randint(1000, 8000), "rating": round(random.uniform(4.0, 5.0), 1)},
            {"name": f"{query} - Budget", "price": random.randint(300, 2000), "rating": round(random.uniform(3.0, 4.5), 1)},
            {"name": f"{query} - Pro", "price": random.randint(2000, 10000), "rating": round(random.uniform(4.0, 5.0), 1)}
        ]
        
        # Filter by max price
        if max_price:
            products = [p for p in products if p["price"] <= max_price]
        
        for i, product in enumerate(products):
            product["id"] = f"PROD{random.randint(10000, 99999)}"
            product["category"] = category or "Electronics"
            product["in_stock"] = random.choice([True, True, True, False])  # 75% in stock
            product["seller"] = random.choice(["Amazon", "Hepsiburada", "Trendyol", "N11"])
        
        return {
            "query": query,
            "category": category,
            "max_price": max_price,
            "results_count": len(products),
            "products": products[:3]  # Return top 3
        }
    
    def _mock_check_calendar(self, date: str, time_range: Optional[str] = None) -> Dict[str, Any]:
        """Mock calendar availability check"""
        # Generate some mock events
        events = []
        num_events = random.randint(0, 4)
        
        for i in range(num_events):
            start_hour = random.randint(9, 17)
            duration = random.choice([1, 2, 3])
            events.append({
                "title": random.choice(["Meeting", "Call", "Review", "Stand-up", "Workshop"]),
                "start_time": f"{start_hour:02d}:00",
                "end_time": f"{start_hour + duration:02d}:00",
                "attendees": random.randint(2, 10)
            })
        
        # Calculate free slots
        all_hours = set(range(9, 18))  # 9 AM to 6 PM
        busy_hours = set()
        for event in events:
            start = int(event["start_time"].split(":")[0])
            end = int(event["end_time"].split(":")[0])
            busy_hours.update(range(start, end))
        
        free_hours = sorted(all_hours - busy_hours)
        free_slots = [f"{h:02d}:00-{h+1:02d}:00" for h in free_hours]
        
        return {
            "date": date,
            "time_range": time_range or "09:00-18:00",
            "events": events,
            "free_slots": free_slots,
            "availability": "available" if len(free_slots) > 0 else "fully_booked"
        }
    
    def _mock_translate_text(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> Dict[str, Any]:
        """Mock translation service"""
        # Simple mock translations for common words
        translations = {
            ("tr", "en"): {
                "elma": "apple",
                "armut": "pear", 
                "kiraz": "cherry",
                "merhaba": "hello",
                "dünya": "world"
            },
            ("en", "tr"): {
                "apple": "elma",
                "pear": "armut",
                "cherry": "kiraz",
                "hello": "merhaba",
                "world": "dünya"
            }
        }
        
        lang_pair = (source_lang or "auto", target_lang)
        translation_dict = translations.get(lang_pair, {})
        translated = translation_dict.get(text.lower(), f"[{text} translated to {target_lang}]")
        
        return {
            "original_text": text,
            "translated_text": translated,
            "source_language": source_lang or "auto-detected",
            "target_language": target_lang,
            "confidence": round(random.uniform(0.85, 0.99), 2)
        }


# Singleton instance
_mock_env_instance = None

def get_mock_environment() -> MockToolEnvironment:
    """Get singleton mock environment instance"""
    global _mock_env_instance
    if _mock_env_instance is None:
        _mock_env_instance = MockToolEnvironment()
    return _mock_env_instance
