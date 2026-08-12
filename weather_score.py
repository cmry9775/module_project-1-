def weather_score(avg_temp_c, rain_chance_pct, humidity_pct):
    #optimal temp, rain_chance, humidity : ISO 7730
    ideal_low, ideal_high = 18, 24
    if ideal_low <= avg_temp_c <= ideal_high:
        temp_score = 100
    else:
        dist = ideal_low - avg_temp_c if avg_temp_c < ideal_low else avg_temp_c - ideal_high
        temp_score = max(0, 100 - dist * 4)

    rain_score = max(0, 100 - rain_chance_pct)

    h_low, h_high = 40, 60
    if h_low <= humidity_pct <= h_high:
        humidity_score = 100
    else:
        dist = h_low - humidity_pct if humidity_pct < h_low else humidity_pct - h_high
        humidity_score = max(0, 100 - dist * 2)

    '''Relative importance (based on |standardized beta|)
    평균기온: 46.8%
    일강수량: 28.6%
    평균상대습도: 24.6%
    '''
    final_score = (
        temp_score * 0.47 +
        rain_score * 0.28 +
        humidity_score * 0.25
    )

    return round(max(1, min(100, final_score)))
