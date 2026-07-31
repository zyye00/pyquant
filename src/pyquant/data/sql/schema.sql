CREATE SCHEMA IF NOT EXISTS ref;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS api;
CREATE SCHEMA IF NOT EXISTS feature;

CREATE TABLE IF NOT EXISTS ref.security (security_id UINTEGER PRIMARY KEY, symbol VARCHAR NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS ref.market_index (index_id USMALLINT PRIMARY KEY, index_code VARCHAR NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS core.stock_daily (security_id UINTEGER NOT NULL, trade_date DATE NOT NULL, open FLOAT, high FLOAT, low FLOAT, close FLOAT, preclose FLOAT, volume BIGINT, amount DOUBLE, turn FLOAT, pe_ttm FLOAT, pb_mrq FLOAT, ps_ttm FLOAT, pcf_ncf_ttm FLOAT, is_st BOOLEAN);
CREATE TABLE IF NOT EXISTS core.index_daily (index_id USMALLINT NOT NULL, trade_date DATE NOT NULL, open FLOAT, high FLOAT, low FLOAT, close DOUBLE, preclose FLOAT, volume BIGINT, amount DOUBLE, turn FLOAT, pe_ttm FLOAT, pb_mrq FLOAT, ps_ttm FLOAT, pcf_ncf_ttm FLOAT, is_st BOOLEAN);
CREATE TABLE IF NOT EXISTS core.index_constituent (index_id USMALLINT NOT NULL, effective_date DATE NOT NULL, security_id UINTEGER NOT NULL, PRIMARY KEY (index_id, effective_date, security_id));
CREATE TABLE IF NOT EXISTS core.dividend (security_id UINTEGER NOT NULL, announce_date DATE, record_date DATE, ex_date DATE, payment_date DATE, cash_dividend_before_tax FLOAT);
CREATE TABLE IF NOT EXISTS core.share_capital_quarterly (security_id UINTEGER NOT NULL, report_date DATE, publish_date DATE, total_shares BIGINT);
CREATE TABLE IF NOT EXISTS meta.stock_daily_coverage (security_id UINTEGER NOT NULL, start_date DATE NOT NULL, end_date DATE NOT NULL, field_set_id UTINYINT NOT NULL, PRIMARY KEY (security_id, start_date, end_date, field_set_id));
CREATE TABLE IF NOT EXISTS meta.index_daily_coverage (index_id USMALLINT NOT NULL, start_date DATE NOT NULL, end_date DATE NOT NULL, field_set_id UTINYINT NOT NULL, PRIMARY KEY (index_id, start_date, end_date, field_set_id));
CREATE TABLE IF NOT EXISTS meta.dividend_coverage (security_id UINTEGER NOT NULL, query_year USMALLINT NOT NULL, field_set_id UTINYINT NOT NULL, PRIMARY KEY (security_id, query_year, field_set_id));
CREATE TABLE IF NOT EXISTS meta.share_capital_coverage (security_id UINTEGER NOT NULL, report_year USMALLINT NOT NULL, report_quarter UTINYINT NOT NULL, PRIMARY KEY (security_id, report_year, report_quarter));
CREATE TABLE IF NOT EXISTS feature.intraday_volatility_daily (security_id UINTEGER NOT NULL, trade_date DATE NOT NULL, vol_daily FLOAT, bar_count USMALLINT, return_count USMALLINT, is_valid BOOLEAN NOT NULL);

CREATE OR REPLACE VIEW api.stock_daily AS SELECT s.symbol, d.trade_date AS date, d.open, d.high, d.low, d.close, d.preclose, d.volume, d.amount, d.turn, 100.0 * (d.close / d.preclose - 1.0) AS pct_chg, d.pe_ttm, d.pb_mrq, d.ps_ttm, d.pcf_ncf_ttm, d.is_st FROM core.stock_daily AS d JOIN ref.security AS s USING (security_id);
CREATE OR REPLACE VIEW api.index_daily AS SELECT d.trade_date AS date, i.index_code AS symbol, d.open, d.high, d.low, d.close, d.preclose, d.volume, d.amount, d.turn, d.close / d.preclose - 1.0 AS pct_chg, d.pe_ttm, d.pb_mrq, d.ps_ttm, d.pcf_ncf_ttm, d.is_st FROM core.index_daily AS d JOIN ref.market_index AS i USING (index_id);
CREATE OR REPLACE VIEW api.index_constituents AS SELECT c.effective_date, i.index_code, s.symbol FROM core.index_constituent AS c JOIN ref.market_index AS i USING (index_id) JOIN ref.security AS s USING (security_id);
CREATE OR REPLACE VIEW api.dividend AS SELECT s.symbol, d.announce_date, d.record_date, d.ex_date AS operate_date, d.payment_date, d.cash_dividend_before_tax FROM core.dividend AS d JOIN ref.security AS s USING (security_id);
CREATE OR REPLACE VIEW api.share_capital_quarterly AS SELECT s.symbol, q.publish_date, q.report_date, q.total_shares FROM core.share_capital_quarterly AS q JOIN ref.security AS s USING (security_id);
CREATE OR REPLACE VIEW api.stock_daily_coverage AS SELECT s.symbol, c.start_date AS start, c.end_date AS end, c.field_set_id FROM meta.stock_daily_coverage AS c JOIN ref.security AS s USING (security_id);
CREATE OR REPLACE VIEW api.dividend_coverage AS SELECT s.symbol, c.query_year AS year FROM meta.dividend_coverage AS c JOIN ref.security AS s USING (security_id) WHERE c.field_set_id = 2;
CREATE OR REPLACE VIEW api.share_capital_coverage AS SELECT s.symbol, c.report_year AS year, c.report_quarter AS quarter FROM meta.share_capital_coverage AS c JOIN ref.security AS s USING (security_id);
CREATE OR REPLACE VIEW api.daily_market_cap AS SELECT s.symbol, m.trade_date, m.close, m.publish_date, m.total_shares, m.close * m.total_shares AS total_market_cap FROM (SELECT p.security_id, p.trade_date, p.close, q.publish_date, q.total_shares FROM core.stock_daily AS p ASOF LEFT JOIN core.share_capital_quarterly AS q ON p.security_id = q.security_id AND p.trade_date >= q.publish_date) AS m JOIN ref.security AS s USING (security_id);
