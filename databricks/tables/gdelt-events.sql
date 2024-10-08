CREATE TABLE IF NOT EXISTS bronze.gdelt_events (
    GLOBALEVENTID BIGINT,
    SQLDATE DATE,
    MonthYear INT,
    Year INT,
    FractionDate FLOAT,
    Actor1Code STRING,
    Actor1Name STRING,
    Actor1CountryCode STRING,
    Actor1KnownGroupCode STRING,
    Actor1EthnicCode STRING,
    Actor1Religion1Code STRING,
    Actor1Religion2Code STRING,
    Actor1Type1Code STRING,
    Actor1Type2Code STRING,
    Actor1Type3Code STRING,
    Actor2Code STRING,
    Actor2Name STRING,
    Actor2CountryCode STRING,
    Actor2KnownGroupCode STRING,
    Actor2EthnicCode STRING,
    Actor2Religion1Code STRING,
    Actor2Religion2Code STRING,
    Actor2Type1Code STRING,
    Actor2Type2Code STRING,
    Actor2Type3Code STRING,
    IsRootEvent INT,
    EventCode STRING,
    EventBaseCode STRING,
    EventRootCode STRING,
    QuadClass INT,
    GoldsteinScale FLOAT,
    NumMentions INT,
    NumSources INT,
    NumArticles INT,
    AvgTone FLOAT,
    Actor1Geo_Type INT,
    Actor1Geo_FullName STRING,
    Actor1Geo_CountryCode STRING,
    Actor1Geo_ADM1Code STRING,
    Actor1Geo_ADM2Code STRING,
    Actor1Geo_Lat FLOAT,
    Actor1Geo_Long FLOAT,
    Actor1Geo_FeatureID STRING,
    Actor2Geo_Type INT,
    Actor2Geo_FullName STRING,
    Actor2Geo_CountryCode STRING,
    Actor2Geo_ADM1Code STRING,
    Actor2Geo_ADM2Code STRING,
    Actor2Geo_Lat FLOAT,
    Actor2Geo_Long FLOAT,
    Actor2Geo_FeatureID STRING,
    ActionGeo_Type INT,
    ActionGeo_FullName STRING,
    ActionGeo_CountryCode STRING,
    ActionGeo_ADM1Code STRING,
    ActionGeo_ADM2Code STRING,
    ActionGeo_Lat FLOAT,
    ActionGeo_Long FLOAT,
    ActionGeo_FeatureID STRING,
    DATEADDED BIGINT,
    SOURCEURL STRING
)
USING DELTA




